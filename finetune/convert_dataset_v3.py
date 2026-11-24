"""
Convert processed bridge inspection QA pairs into SFT JSONL format (v3).

Differences from v2:
  - Source: $BRIDGE_ROOT/pdf_processed_ALL_vermont_reports (full 3220-scene set)
  - Hold out the 200 BridgeEQA test scenes (listed in splits/test_200_scenes.txt)
  - Drop 78 problem scenes flagged by audit_data.py (no_qa / missing core)
  - Split remaining 95/5 train/val (VAL_RATIO=0.05, seed 42)
  - Filter hallucinated image references (resolved filename not in scene_graph.nodes)
  - Keeps QAs with no rating / no reference_images (per user decision)

Input  : $BRIDGE_ROOT/pdf_processed_ALL_vermont_reports/<scene>/
              qa_pairs.json, scene_graph.json, image_reference_mappings.json, images/
Output : $BRIDGE_ROOT/bridge_finetune/data_v3/train.jsonl, val.jsonl, stats.json

The paper corpora were written with:
  --variant_mode random --output_dir .../data_v3_mixed   (Single-Context, SC)
  --variant_mode all    --output_dir .../data_v3_all     (Multi-Context, MC)
BRIDGE_ROOT defaults to /home/projects. Note that image paths written into the
JSONL are absolute, so the training job must see the same filesystem layout.
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

BRIDGE_ROOT = Path(os.environ.get("BRIDGE_ROOT", "/home/projects"))
_REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = BRIDGE_ROOT / "pdf_processed_ALL_vermont_reports"
OUTPUT_DIR = BRIDGE_ROOT / "bridge_finetune" / "data_v3"
TEST_LIST = _REPO / "splits" / "test_200_scenes.txt"
AUDIT_FLAGS_CSV = _REPO / "finetune" / "audit_output" / "scene_flags.csv"
VAL_RATIO = 0.05
RANDOM_SEED = 42
VARIANT_MODE = "all"  # "all" = 3 variants per QA; "random" = 1 random variant per QA
MAX_IMAGES_PER_SAMPLE = 8
HASH_RE = re.compile(r"^[0-9a-f]{32}\.png$")
CITATION_RE = re.compile(r"\[img:([^\]]+)\]")

SYSTEM_PROMPT = (
    "You are an expert bridge inspector trained on National Bridge Inspection "
    "Standards (NBIS). Given inspection evidence (photographs, scene graph "
    "descriptions, or both), answer the question as a JSON object with these fields:\n\n"
    '- "answer": Your findings with inline image citations using [img:filename] '
    "format wherever a specific image supports a claim.\n"
    '- "reference_images": Array of all cited image filenames (deduplicated).\n'
    '- "condition_rating": {"score": N} where N is the NBI 0-9 scale rating, '
    "or null if not applicable.\n\n"
    "NBI scale: 9=Excellent, 8=Very Good, 7=Good, 6=Satisfactory, 5=Fair, "
    "4=Poor, 3=Serious, 2=Critical, 1=Imminent Failure, 0=Failed."
)


# ── Citation helpers ─────────────────────────────────────────────────────────

def extract_inline_citations(text: str) -> list[str]:
    return CITATION_RE.findall(text)


def validate_citations(answer: str, reference_images: list[str]) -> bool:
    cited = set(extract_inline_citations(answer))
    return cited.issubset(set(reference_images))


def inject_inline_citations(answer: str, reference_images: list[str]) -> str:
    if not reference_images:
        return answer
    existing = set(extract_inline_citations(answer))
    new_refs = [img for img in reference_images if img not in existing]
    if not new_refs:
        return answer
    tags = " ".join(f"[img:{img}]" for img in new_refs)
    return f"{answer}\n\nReferenced evidence: {tags}"


# ── Scene graph context builder ──────────────────────────────────────────────

def build_sg_context_text(nodes: list[dict], relevant_images: list[str]) -> str:
    name_to_idx = {n["image_name"]: i for i, n in enumerate(nodes) if n.get("image_name")}
    relevant_idx = {name_to_idx[img] for img in relevant_images if img in name_to_idx}
    neighbor_idx = set()
    for idx in relevant_idx:
        for edge in nodes[idx].get("edges", []):
            t = edge.get("connected_to", "")
            if t in name_to_idx:
                neighbor_idx.add(name_to_idx[t])
    included = sorted(relevant_idx | neighbor_idx)

    lines = ["## Scene Graph"]
    for idx in included:
        n = nodes[idx]
        lines.append(f"Node {idx} [{n['image_name']}]: {n.get('central_focus','')}. {n.get('image_description','')}")
    edge_lines = []
    for idx in included:
        for edge in nodes[idx].get("edges", []):
            t = edge.get("connected_to", "")
            if t in name_to_idx and name_to_idx[t] in (relevant_idx | neighbor_idx):
                edge_lines.append(f"Node {idx} → Node {name_to_idx[t]}: {edge.get('description_of_connection','')}")
    if edge_lines:
        lines.append("")
        lines.append("## Edges")
        lines.extend(edge_lines)
    return "\n".join(lines)


# ── User content builders ────────────────────────────────────────────────────

def _suffix(require_rating: bool) -> str:
    return " A numerical value for the condition rating is required." if require_rating \
           else " A numerical value for the condition rating is optional."


def build_user_images(q, n_imgs, req): return f"{'<image>'*n_imgs}{q}{_suffix(req)}"
def build_user_sg(q, sg, req):         return f"{sg}\n\n{q}{_suffix(req)}"
def build_user_both(q, n_imgs, sg, req): return f"{'<image>'*n_imgs}\n{sg}\n\n{q}{_suffix(req)}"


def build_assistant_json(answer, reference_images, condition_rating) -> str:
    return json.dumps({
        "answer": answer,
        "reference_images": reference_images,
        "condition_rating": condition_rating,
    }, ensure_ascii=False)


# ── Image resolution + hallucination filter ─────────────────────────────────

def build_label_to_images(mappings: list[dict]) -> dict[str, list[str]]:
    m: dict[str, list[str]] = defaultdict(list)
    for entry in mappings:
        label = (entry.get("reference_images") or "").strip()
        fname = (entry.get("image_name") or "").strip()
        if label and fname and fname not in m[label]:
            m[label].append(fname)
    return dict(m)


def resolve_reference_images(
    refs: list[str],
    label_map: dict[str, list[str]],
    images_dir: Path,
    node_imgs: set[str],
) -> tuple[list[str], int]:
    """Resolve refs to filenames; filter hallucinations (not in scene graph).

    Returns (resolved_filenames, n_hallucinations_dropped).
    """
    resolved: list[str] = []
    seen: set[str] = set()
    dropped_hallucinated = 0

    for ref in refs:
        ref = ref.strip()
        candidates = [ref] if HASH_RE.match(ref) else label_map.get(ref, [])
        if not candidates:
            dropped_hallucinated += 1
            continue
        for fname in candidates:
            if fname in seen:
                continue
            # must exist on disk AND be in scene graph (to filter hallucinations)
            if not (images_dir / fname).exists():
                dropped_hallucinated += 1
                continue
            if node_imgs and fname not in node_imgs:
                dropped_hallucinated += 1
                continue
            resolved.append(fname)
            seen.add(fname)

    return resolved[:MAX_IMAGES_PER_SAMPLE], dropped_hallucinated


# ── Per-bridge conversion ────────────────────────────────────────────────────

def convert_bridge(bridge_dir: Path, variant_mode: str = "all", rng: Optional[random.Random] = None) -> tuple[list[dict], dict]:
    qa_file = bridge_dir / "qa_pairs.json"
    mapping_file = bridge_dir / "image_reference_mappings.json"
    sg_file = bridge_dir / "scene_graph.json"
    images_dir = bridge_dir / "images"

    stats = {
        "bridge": bridge_dir.name,
        "qa_total": 0,
        "qa_converted": 0,
        "skipped_bad_citations": 0,
        "variants_produced": 0,
        "hallucinated_refs_filtered": 0,
        "qas_left_without_refs_after_filter": 0,
    }
    if not qa_file.exists():
        return [], stats

    qa_pairs = json.loads(qa_file.read_text())
    stats["qa_total"] = len(qa_pairs)

    label_map = build_label_to_images(json.loads(mapping_file.read_text())) if mapping_file.exists() else {}
    sg_nodes = json.loads(sg_file.read_text()).get("nodes", []) if sg_file.exists() else []
    node_imgs = {n.get("image_name") for n in sg_nodes if n.get("image_name")}

    samples: list[dict] = []
    for qa in qa_pairs:
        question = (qa.get("question") or "").strip()
        answer = (qa.get("answer") or "").strip()
        refs = qa.get("reference_images") or []
        cr = qa.get("condition_rating")

        if not question or not answer:
            continue

        resolved, n_halluc = resolve_reference_images(refs, label_map, images_dir, node_imgs)
        stats["hallucinated_refs_filtered"] += n_halluc
        if refs and not resolved:
            stats["qas_left_without_refs_after_filter"] += 1
            # keep the QA (per user), just with empty reference_images

        # Strip inline [img:X] citations that reference hallucinated images
        cited_inline = set(extract_inline_citations(answer))
        bad_inline = cited_inline - set(resolved)
        if bad_inline:
            for bad in bad_inline:
                answer = answer.replace(f"[img:{bad}]", "").strip()

        if not validate_citations(answer, resolved):
            stats["skipped_bad_citations"] += 1
            continue

        has_rating = cr is not None and cr.get("score") is not None
        answer_with_citations = inject_inline_citations(answer, resolved)
        image_paths = [str((images_dir / f).resolve()) for f in resolved]

        assistant_json = build_assistant_json(answer_with_citations, resolved, cr)
        sg_text = build_sg_context_text(sg_nodes, resolved) if sg_nodes else ""

        variant_samples = {}
        variant_samples["images"] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_images(question, len(resolved), has_rating)},
                {"role": "assistant", "content": assistant_json},
            ],
            "images": image_paths,
        }
        if sg_text:
            variant_samples["sg_context"] = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_sg(question, sg_text, has_rating)},
                    {"role": "assistant", "content": assistant_json},
                ],
                "images": [],
            }
            variant_samples["both"] = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_both(question, len(resolved), sg_text, has_rating)},
                    {"role": "assistant", "content": assistant_json},
                ],
                "images": image_paths,
            }

        if variant_mode == "random":
            choice = (rng or random).choice(list(variant_samples.keys()))
            samples.append(variant_samples[choice])
            stats.setdefault("variant_picked", {}).setdefault(choice, 0)
            stats["variant_picked"][choice] += 1
            stats["variants_produced"] += 1
        else:
            for s in variant_samples.values():
                samples.append(s)
            stats["variants_produced"] += len(variant_samples)
        stats["qa_converted"] += 1

    return samples, stats


# ── Scene selection ──────────────────────────────────────────────────────────

def load_excluded_scenes() -> tuple[set[str], set[str]]:
    test_scenes = {s.strip() for s in TEST_LIST.read_text().splitlines() if s.strip()}
    import csv
    drop_scenes: set[str] = set()
    with AUDIT_FLAGS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("recommend_drop", "").strip().lower() == "true":
                drop_scenes.add(row["scene"])
    return test_scenes, drop_scenes


def split_bridges(bridge_dirs, val_ratio, seed):
    rng = random.Random(seed)
    dirs = sorted(bridge_dirs)
    rng.shuffle(dirs)
    n_val = max(1, round(len(dirs) * val_ratio))
    return dirs[n_val:], dirs[:n_val]


def write_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    test_scenes, drop_scenes = load_excluded_scenes()
    print(f"Test hold-out: {len(test_scenes)} scenes")
    print(f"Audit-drop:    {len(drop_scenes)} scenes")
    overlap = test_scenes & drop_scenes
    print(f"  (of which {len(overlap)} overlap with test — use BridgeEQA version for test)")
    excluded = test_scenes | drop_scenes

    all_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    eligible = [d for d in all_dirs if d.name not in excluded and (d / "qa_pairs.json").exists()]
    print(f"Eligible scenes (all - test - audit-drop): {len(eligible)} / {len(all_dirs)}")

    train_dirs, val_dirs = split_bridges(eligible, VAL_RATIO, RANDOM_SEED)
    print(f"Train scenes: {len(train_dirs)}   Val scenes: {len(val_dirs)}  (val_ratio={VAL_RATIO})")

    variant_mode = args.variant_mode
    print(f"Variant mode: {variant_mode}")
    rng_variants = random.Random(RANDOM_SEED + 1)

    train_samples, val_samples, all_stats = [], [], []
    print("\nConverting training scenes...")
    for i, bdir in enumerate(train_dirs, 1):
        s, st = convert_bridge(bdir, variant_mode=variant_mode, rng=rng_variants)
        train_samples.extend(s)
        all_stats.append({**st, "split": "train"})
        if i % 200 == 0 or i == len(train_dirs):
            print(f"  [{i}/{len(train_dirs)}] train samples so far: {len(train_samples)}")

    print("\nConverting val scenes...")
    for i, bdir in enumerate(val_dirs, 1):
        s, st = convert_bridge(bdir, variant_mode=variant_mode, rng=rng_variants)
        val_samples.extend(s)
        all_stats.append({**st, "split": "val"})
        if i % 50 == 0 or i == len(val_dirs):
            print(f"  [{i}/{len(val_dirs)}] val samples so far: {len(val_samples)}")

    write_jsonl(train_samples, output_dir / "train.jsonl")
    write_jsonl(val_samples, output_dir / "val.jsonl")

    def agg(ss):
        return {
            "bridges": len(ss),
            "qa_total": sum(s["qa_total"] for s in ss),
            "qa_converted": sum(s["qa_converted"] for s in ss),
            "variants_produced": sum(s["variants_produced"] for s in ss),
            "skipped_bad_citations": sum(s["skipped_bad_citations"] for s in ss),
            "hallucinated_refs_filtered": sum(s["hallucinated_refs_filtered"] for s in ss),
            "qas_left_without_refs_after_filter": sum(s["qas_left_without_refs_after_filter"] for s in ss),
        }

    summary = {
        "source_dataset": str(dataset_dir),
        "test_holdout": sorted(test_scenes),
        "audit_dropped": sorted(drop_scenes),
        "train": agg([s for s in all_stats if s["split"] == "train"]),
        "val": agg([s for s in all_stats if s["split"] == "val"]),
        "per_bridge": all_stats,
    }
    (output_dir / "stats.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    for split, s, path in [("TRAIN", summary["train"], output_dir / "train.jsonl"),
                            ("VAL",   summary["val"],   output_dir / "val.jsonl")]:
        print(f"\n{split} -> {path}")
        for k, v in s.items():
            print(f"  {k:<38} {v}")
    print(f"\nStats: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=str(DATASET_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--variant_mode", choices=["all", "random"], default=VARIANT_MODE,
                        help="'all' keeps 3 variants per QA; 'random' picks 1 at random")
    args = parser.parse_args()
    main(args)
