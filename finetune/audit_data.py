"""Data-quality audit of $BRIDGE_ROOT/pdf_processed_ALL_vermont_reports (BRIDGE_ROOT defaults to /home/projects).

Flags scenes with structural problems and writes:
  audit_output/issues_per_scene.csv   one row per (scene, issue) flag
  audit_output/scene_flags.csv        one row per scene with all flags as columns
  audit_output/summary.md             human-readable report

Run: conda activate bridge_ft && python audit_data.py
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

BASE = Path(os.environ.get("BRIDGE_ROOT", "/home/projects")) / "pdf_processed_ALL_vermont_reports"
OUT = Path(__file__).resolve().parent / "audit_output"
OUT.mkdir(exist_ok=True)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def audit_scene(d: Path) -> dict:
    flags = {"scene": d.name}
    cr = _load(d / "condition_ratings.json")
    qa = _load(d / "qa_pairs.json")
    sg = _load(d / "scene_graph.json")
    irm = _load(d / "image_reference_mappings.json")

    # file presence
    flags["missing_condition_ratings"] = cr is None
    flags["missing_qa_pairs"] = qa is None
    flags["missing_scene_graph"] = sg is None
    flags["missing_image_reference_mappings"] = irm is None

    # image files on disk
    img_dir = d / "images"
    img_files = set()
    if img_dir.exists():
        img_files = {f.name for f in img_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    flags["n_img_files"] = len(img_files)
    flags["no_image_files"] = len(img_files) == 0

    # scene graph
    nodes = (sg or {}).get("nodes", []) if sg else []
    node_imgs = {n.get("image_name") for n in nodes if n.get("image_name")}
    flags["n_nodes"] = len(nodes)
    flags["no_nodes"] = len(nodes) == 0
    flags["single_node"] = len(nodes) == 1

    # missing image files (cited in sg but not on disk)
    missing_on_disk = node_imgs - img_files
    flags["n_graph_imgs_missing_on_disk"] = len(missing_on_disk)
    flags["graph_imgs_missing_on_disk"] = len(missing_on_disk) > 0

    # QA
    qa_list = qa or []
    flags["n_qa"] = len(qa_list)
    flags["no_qa"] = len(qa_list) == 0
    flags["short_qa_list"] = 0 < len(qa_list) < 10
    n_no_rating = 0
    n_no_refimgs = 0
    n_invalid_rating = 0
    n_short_answer = 0
    n_identity = 0
    n_refimg_not_in_graph = 0
    total_refs = 0
    cited_imgs = set()
    for q in qa_list:
        sc = (q.get("condition_rating") or {}).get("score")
        if not isinstance(sc, (int, float)):
            n_no_rating += 1
        elif sc < 0 or sc > 9:
            n_invalid_rating += 1
        refs = q.get("reference_images") or []
        if not refs:
            n_no_refimgs += 1
        total_refs += len(refs)
        for r in refs:
            cited_imgs.add(r)
            if r not in node_imgs:
                n_refimg_not_in_graph += 1
        ans = (q.get("answer") or "").strip()
        if len(ans) < 20:
            n_short_answer += 1
        if ans and ans == (q.get("question") or "").strip():
            n_identity += 1

    flags["n_qa_no_rating"] = n_no_rating
    flags["n_qa_no_refimgs"] = n_no_refimgs
    flags["n_qa_invalid_rating"] = n_invalid_rating
    flags["n_qa_short_answer"] = n_short_answer
    flags["n_qa_identity"] = n_identity
    flags["n_refimg_not_in_graph"] = n_refimg_not_in_graph
    flags["any_refimg_not_in_graph"] = n_refimg_not_in_graph > 0
    flags["frac_qa_without_rating"] = n_no_rating / len(qa_list) if qa_list else float("nan")
    flags["frac_qa_without_refimgs"] = n_no_refimgs / len(qa_list) if qa_list else float("nan")

    # condition ratings
    cr_list = (cr or {}).get("condition_ratings", []) if cr else []
    el_list = (cr or {}).get("element_ratings", []) if cr else []
    flags["n_components"] = len(cr_list)
    flags["no_components"] = len(cr_list) == 0
    flags["n_elements"] = len(el_list)

    return flags


def main():
    dirs = sorted([d for d in BASE.glob("BridgeInspRpt-*") if d.is_dir()])
    print(f"Auditing {len(dirs)} scenes ...")
    rows = [audit_scene(d) for d in dirs]
    df = pd.DataFrame(rows)

    # Flag "drop" recommendation: any critical problem
    df["drop_reason"] = ""
    reasons = []
    for i, r in df.iterrows():
        why = []
        if r["missing_condition_ratings"] or r["missing_qa_pairs"] or r["missing_scene_graph"]:
            why.append("missing_core_file")
        if r["no_qa"]:
            why.append("no_qa")
        if r["no_nodes"] or r["no_image_files"]:
            why.append("no_images")
        if r["no_components"]:
            why.append("no_components")
        if r["graph_imgs_missing_on_disk"]:
            why.append("missing_image_files")
        if r["frac_qa_without_rating"] == 1.0:
            why.append("no_rated_qa")
        reasons.append(",".join(why))
    df["drop_reason"] = reasons
    df["recommend_drop"] = df["drop_reason"] != ""

    df.to_csv(OUT / "scene_flags.csv", index=False)

    # Long-form issue list
    issue_cols = [
        "missing_condition_ratings", "missing_qa_pairs", "missing_scene_graph",
        "missing_image_reference_mappings", "no_image_files", "no_nodes", "single_node",
        "graph_imgs_missing_on_disk", "no_qa", "short_qa_list", "any_refimg_not_in_graph",
        "no_components",
    ]
    long = []
    for _, r in df.iterrows():
        for c in issue_cols:
            if r[c]:
                long.append({"scene": r["scene"], "issue": c})
    pd.DataFrame(long).to_csv(OUT / "issues_per_scene.csv", index=False)

    # Summary
    total = len(df)
    to_drop = int(df["recommend_drop"].sum())
    print(f"\nScenes: {total}")
    print(f"Recommended drop: {to_drop}")
    print("\n--- Issue counts ---")
    for c in issue_cols:
        n = int(df[c].sum())
        if n: print(f"  {c:<45} {n:>4}  ({n/total:.1%})")
    print("\n--- QA-level problem counts (aggregate) ---")
    for c in ["n_qa_no_rating", "n_qa_no_refimgs", "n_qa_invalid_rating",
              "n_qa_short_answer", "n_qa_identity", "n_refimg_not_in_graph"]:
        print(f"  {c:<28} {int(df[c].sum()):>6}")

    # Summary markdown
    lines = []
    lines.append(f"# Data-Quality Audit\n")
    lines.append(f"Scenes: **{total}**. Recommended drop: **{to_drop}** ({to_drop/total:.1%})\n")
    lines.append("\n## Issue counts (per scene)\n")
    lines.append("| issue | count | % |")
    lines.append("|---|---|---|")
    for c in issue_cols:
        n = int(df[c].sum())
        if n: lines.append(f"| {c} | {n} | {n/total:.1%} |")
    lines.append("\n## QA-level problem counts (sum across all 63k QAs)\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for c in ["n_qa_no_rating", "n_qa_no_refimgs", "n_qa_invalid_rating",
              "n_qa_short_answer", "n_qa_identity", "n_refimg_not_in_graph"]:
        lines.append(f"| {c} | {int(df[c].sum())} |")
    lines.append("\n## Drop reasons breakdown\n")
    cnt = Counter()
    for r in df["drop_reason"]:
        if r:
            for x in r.split(","):
                cnt[x] += 1
    for k, v in cnt.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Sample dropped scenes (first 20)\n")
    for s in df[df["recommend_drop"]].head(20)["scene"].tolist():
        lines.append(f"- {s}  ({df.loc[df['scene']==s, 'drop_reason'].iloc[0]})")
    (OUT / "summary.md").write_text("\n".join(lines))

    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
