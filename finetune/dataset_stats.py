"""Statistical analysis of processed Vermont bridge inspection reports.

Scans $BRIDGE_ROOT/pdf_processed_ALL_vermont_reports (default /home/projects), reads per-scene JSON
outputs (condition_ratings, qa_pairs, scene_graph, image_reference_mappings),
and writes:

  stats_output/
    per_scene.csv         one row per scene with summary metrics
    component_ratings.csv one row per (scene, component, score)
    element_ratings.csv   one row per (scene, element, score) — CS scale 1-4
    qa_pairs_flat.csv     one row per QA
    summary.md            human-readable report
    distributions.json    machine-readable histograms
    plots/                *.png figures

Run: conda activate bridge_ft && python dataset_stats.py
"""

import json
import os
import re
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("BRIDGE_ROOT", "/home/projects")) / "pdf_processed_ALL_vermont_reports"
OUT = Path(__file__).resolve().parent / "stats_output"
PLOTS = OUT / "plots"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# ── Agency normalization ──────────────────────────────────────────────────────

def normalize_agency(raw: str) -> str:
    if not raw:
        return "Unknown"
    s = raw.strip()
    sl = s.lower()
    if "private" in sl and "railroad" not in sl:
        return "Private"
    if "railroad" in sl:
        return "Railroad"
    if "state" in sl or sl == "vermont" or sl.startswith("s01"):
        return "State"
    if "city" in sl or "municipal" in sl or sl.startswith("l03"):
        return "City/Municipal"
    if "village" in sl:
        return "Village"
    if "town" in sl or sl.startswith("l02"):
        return "Town/Township"
    return "Other"


# ── Load ───────────────────────────────────────────────────────────────────────

def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def scan():
    dirs = sorted([d for d in BASE.glob("BridgeInspRpt-*") if d.is_dir()])
    scenes = []
    for d in dirs:
        cr = _load(d / "condition_ratings.json")
        qa = _load(d / "qa_pairs.json")
        sg = _load(d / "scene_graph.json")
        irm = _load(d / "image_reference_mappings.json")
        n_img_files = len(list((d / "images").glob("*.png"))) + len(list((d / "images").glob("*.jpg")))
        scenes.append({
            "scene": d.name,
            "path": d,
            "cr": cr, "qa": qa, "sg": sg, "irm": irm,
            "n_img_files": n_img_files,
        })
    return scenes


# ── Extract per-scene and flat records ────────────────────────────────────────

def build_per_scene_df(scenes):
    rows = []
    for s in scenes:
        cr, qa, sg = s["cr"], s["qa"], s["sg"]
        town_m = re.match(r"BridgeInspRpt-([A-Z0-9]+)-", s["scene"])
        town = town_m.group(1) if town_m else ""
        nodes = (sg or {}).get("nodes", [])
        qa_list = qa or []
        cr_list = (cr or {}).get("condition_ratings", [])
        el_list = (cr or {}).get("element_ratings", [])
        cited_imgs = set()
        for q in qa_list:
            for img in (q.get("reference_images") or []):
                cited_imgs.add(img)
        node_imgs = {n.get("image_name") for n in nodes if n.get("image_name")}
        cited_in_scene = len(cited_imgs & node_imgs)
        qa_with_rating = sum(
            1 for q in qa_list if isinstance((q.get("condition_rating") or {}).get("score"), (int, float))
        )
        qa_with_cite = sum(1 for q in qa_list if q.get("reference_images"))
        insp_date = (cr or {}).get("inspection_date") or ""
        rows.append({
            "scene": s["scene"],
            "town": town,
            "agency_raw": (cr or {}).get("agency", ""),
            "agency": normalize_agency((cr or {}).get("agency", "")),
            "inspection_type": (cr or {}).get("inspection_type", ""),
            "inspection_date": insp_date,
            "inspection_year": insp_date[:4] if len(insp_date) >= 4 else "",
            "bin": (cr or {}).get("bin", ""),
            "n_nodes": len(nodes),
            "n_img_files": s["n_img_files"],
            "n_qa": len(qa_list),
            "n_qa_with_rating": qa_with_rating,
            "n_qa_with_cite": qa_with_cite,
            "n_components": len(cr_list),
            "n_elements": len(el_list),
            "n_nodes_cited": cited_in_scene,
            "frac_nodes_cited": cited_in_scene / len(node_imgs) if node_imgs else float("nan"),
        })
    return pd.DataFrame(rows)


def build_component_df(scenes):
    rows = []
    for s in scenes:
        cr = s["cr"] or {}
        for c in cr.get("condition_ratings", []):
            score = (c.get("component_rating") or {}).get("score")
            if not isinstance(score, (int, float)):
                continue
            rows.append({
                "scene": s["scene"],
                "agency": normalize_agency(cr.get("agency", "")),
                "inspection_year": (cr.get("inspection_date") or "")[:4],
                "component": c.get("component_name", ""),
                "score": int(score),
                "observation": (c.get("condition_observation") or {}).get("observation", ""),
            })
    return pd.DataFrame(rows)


def build_element_df(scenes):
    rows = []
    for s in scenes:
        cr = s["cr"] or {}
        for e in cr.get("element_ratings", []):
            score = (e.get("element_rating") or {}).get("score")
            if not isinstance(score, (int, float)):
                continue
            rows.append({
                "scene": s["scene"],
                "agency": normalize_agency(cr.get("agency", "")),
                "inspection_year": (cr.get("inspection_date") or "")[:4],
                "element": e.get("element_name", ""),
                "score": int(score),
            })
    return pd.DataFrame(rows)


def build_qa_df(scenes):
    rows = []
    for s in scenes:
        for q in (s["qa"] or []):
            sc = (q.get("condition_rating") or {}).get("score")
            refs = q.get("reference_images") or []
            rows.append({
                "scene": s["scene"],
                "question_len": len(q.get("question", "")),
                "answer_len": len(q.get("answer", "")),
                "rating": int(sc) if isinstance(sc, (int, float)) else None,
                "n_images_cited": len(refs),
            })
    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def savefig(name: str):
    p = PLOTS / name
    plt.tight_layout()
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    return p


def hist(data, title, xlabel, fname, bins=30, **kw):
    plt.figure(figsize=(7, 4))
    plt.hist(data, bins=bins, color="#3b82f6", edgecolor="white", **kw)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.grid(axis="y", alpha=0.3)
    return savefig(fname)


def bar(series, title, xlabel, fname, top=20, horizontal=False, rotation=30):
    plt.figure(figsize=(8, 4.5) if not horizontal else (7, max(3, 0.3 * min(len(series), top))))
    s = series.head(top)
    if horizontal:
        plt.barh(s.index[::-1], s.values[::-1], color="#10b981")
        plt.xlabel("count")
    else:
        plt.bar(s.index, s.values, color="#10b981")
        plt.xticks(rotation=rotation, ha="right")
        plt.ylabel("count")
    plt.title(title); plt.xlabel(xlabel) if not horizontal else None
    plt.grid(axis="x" if horizontal else "y", alpha=0.3)
    return savefig(fname)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Scanning {BASE} ...")
    scenes = scan()
    print(f"Found {len(scenes)} scenes")

    df_scene = build_per_scene_df(scenes)
    df_comp = build_component_df(scenes)
    df_elem = build_element_df(scenes)
    df_qa = build_qa_df(scenes)

    # Save CSVs
    df_scene.drop(columns=["path"], errors="ignore").to_csv(OUT / "per_scene.csv", index=False)
    df_comp.to_csv(OUT / "component_ratings.csv", index=False)
    df_elem.to_csv(OUT / "element_ratings.csv", index=False)
    df_qa.to_csv(OUT / "qa_pairs_flat.csv", index=False)
    print("CSVs written.")

    # ── Plots ──
    # Scene-level histograms
    hist(df_scene["n_nodes"], "Images per scene (scene_graph nodes)", "# images", "scene_images_hist.png", bins=40)
    hist(df_scene["n_qa"], "QA pairs per scene", "# QA pairs", "qa_per_scene_hist.png", bins=range(0, 25))
    hist(df_scene["n_components"], "Components rated per scene", "# components", "components_per_scene_hist.png", bins=range(0, 13))
    hist(df_scene["n_elements"], "Elements rated per scene", "# elements", "elements_per_scene_hist.png", bins=range(0, 60))
    hist(df_scene["frac_nodes_cited"].dropna(), "Fraction of images cited per scene", "fraction cited", "frac_cited_hist.png", bins=np.linspace(0, 1, 21))

    # Temporal
    yr = df_scene["inspection_year"].replace("", np.nan).dropna().value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(yr.index.astype(str), yr.values, color="#f59e0b")
    plt.title("Scenes by inspection year"); plt.xlabel("year"); plt.ylabel("count")
    plt.grid(axis="y", alpha=0.3); savefig("scenes_by_year.png")

    # Agency (normalized)
    ag = df_scene["agency"].value_counts()
    bar(ag, "Scenes by agency (normalized)", "agency", "scenes_by_agency.png", top=10, horizontal=True)

    # Top towns
    tw = df_scene["town"].value_counts()
    bar(tw, "Top 20 towns", "town", "top_towns.png", top=20)

    # Component ratings
    # Per component boxplot
    top_components = df_comp["component"].value_counts().head(8).index.tolist()
    plt.figure(figsize=(8, 4.5))
    data = [df_comp[df_comp["component"] == c]["score"].values for c in top_components]
    plt.boxplot(data, labels=top_components, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#93c5fd"))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("condition score (0-9)")
    plt.title("Component condition-rating distribution by component")
    plt.grid(axis="y", alpha=0.3)
    savefig("component_rating_box.png")

    # Stacked histogram for top components
    plt.figure(figsize=(8, 4.5))
    width = 0.12
    xs = np.arange(10)
    for i, comp in enumerate(top_components[:6]):
        cnt = df_comp[df_comp["component"] == comp]["score"].value_counts().reindex(xs, fill_value=0)
        plt.bar(xs + i * width, cnt.values, width=width, label=comp)
    plt.xticks(xs + 3 * width, xs)
    plt.xlabel("condition score (0-9)"); plt.ylabel("count")
    plt.title("Score distribution per component (top 6)")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3)
    savefig("component_score_distribution.png")

    # Component rating by agency (heatmap)
    pivot = (df_comp.groupby(["agency", "component"])["score"].mean()
             .unstack().reindex(columns=top_components))
    plt.figure(figsize=(8, 3.5))
    im = plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=4, vmax=8)
    plt.colorbar(im, label="mean score")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(top_components)), top_components, rotation=25, ha="right")
    plt.title("Mean component rating by agency")
    savefig("component_rating_heatmap.png")

    # Element ratings (CS scale, lower=better)
    top_elements = df_elem["element"].value_counts().head(10).index.tolist()
    plt.figure(figsize=(9, 5))
    data = [df_elem[df_elem["element"] == el]["score"].values for el in top_elements]
    plt.boxplot(data, labels=top_elements, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#fca5a5"))
    plt.xlabel("condition state (1=Good, 4=Severe)")
    plt.title("Top 10 elements — CS-scale rating distribution")
    plt.grid(axis="x", alpha=0.3)
    savefig("element_rating_box.png")

    # QA-level
    hist(df_qa["rating"].dropna(), "QA-level rating distribution", "rating (0-9)",
         "qa_rating_hist.png", bins=np.arange(-0.5, 10.5, 1))
    hist(df_qa["n_images_cited"], "Images cited per QA", "# images cited",
         "qa_images_cited_hist.png", bins=range(0, 15))
    hist(df_qa["answer_len"].clip(upper=2500), "Answer length (chars, clipped at 2500)",
         "length (chars)", "qa_answer_len_hist.png", bins=40)

    # Component rating vs number of images per scene — scatter
    scene_mean_rating = df_comp.groupby("scene")["score"].mean().rename("mean_rating")
    merged = df_scene.set_index("scene").join(scene_mean_rating)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(merged["n_nodes"], merged["mean_rating"], s=6, alpha=0.3, color="#6366f1")
    plt.xlabel("# images per scene"); plt.ylabel("mean component rating")
    plt.title("Images per scene vs. mean component rating")
    plt.grid(alpha=0.3); savefig("images_vs_rating.png")

    # ── Distributions JSON ──
    distributions = {
        "n_scenes": int(len(df_scene)),
        "n_qa_total": int(df_qa.shape[0]),
        "qa_with_rating_frac": float(df_qa["rating"].notna().mean()),
        "qa_with_cite_frac": float((df_qa["n_images_cited"] > 0).mean()),
        "images_per_scene": df_scene["n_nodes"].describe().to_dict(),
        "qa_per_scene": df_scene["n_qa"].describe().to_dict(),
        "elements_per_scene": df_scene["n_elements"].describe().to_dict(),
        "components_per_scene": df_scene["n_components"].describe().to_dict(),
        "agency_counts": df_scene["agency"].value_counts().to_dict(),
        "year_counts": df_scene["inspection_year"].value_counts().to_dict(),
        "component_means": df_comp.groupby("component")["score"].agg(["count", "mean", "median", "std"]).round(3).to_dict(),
        "element_means_top15": df_elem.groupby("element")["score"].agg(["count", "mean", "median"]).round(3)
            .sort_values("count", ascending=False).head(15).to_dict(),
        "qa_rating_hist": df_qa["rating"].dropna().astype(int).value_counts().sort_index().to_dict(),
    }
    (OUT / "distributions.json").write_text(json.dumps(distributions, indent=2, default=str))

    # ── Summary markdown ──
    lines = []
    lines.append(f"# Dataset Statistical Summary\n")
    lines.append(f"Scenes scanned: **{len(df_scene)}**\n")
    lines.append(f"Total QA pairs: **{len(df_qa):,}** (rated: {df_qa['rating'].notna().sum():,}, with image cite: {(df_qa['n_images_cited']>0).sum():,})\n")
    lines.append("\n## Scene size\n")
    lines.append(df_scene[["n_nodes", "n_qa", "n_components", "n_elements"]].describe().round(2).to_markdown())
    lines.append("\n\n## Agency (normalized)\n")
    lines.append(df_scene["agency"].value_counts().to_frame("count").to_markdown())
    lines.append("\n\n## Inspection year\n")
    lines.append(df_scene["inspection_year"].value_counts().sort_index().to_frame("count").to_markdown())
    lines.append("\n\n## Component ratings (0-9 scale, higher=better)\n")
    comp_stats = df_comp.groupby("component")["score"].agg(["count", "mean", "median", "std"]).round(2).sort_values("count", ascending=False)
    lines.append(comp_stats.to_markdown())
    lines.append("\n\n## Element ratings top 15 (CS scale 1-4, lower=better)\n")
    elem_stats = df_elem.groupby("element")["score"].agg(["count", "mean", "median"]).round(2).sort_values("count", ascending=False).head(15)
    lines.append(elem_stats.to_markdown())
    lines.append("\n\n## QA-level\n")
    lines.append(df_qa[["rating", "n_images_cited", "answer_len"]].describe().round(2).to_markdown())
    lines.append("\n\n## Top 20 towns\n")
    lines.append(df_scene["town"].value_counts().head(20).to_frame("count").to_markdown())
    lines.append("\n\n## Plots\n")
    for p in sorted(PLOTS.glob("*.png")):
        lines.append(f"- ![{p.stem}](plots/{p.name})")
    (OUT / "summary.md").write_text("\n".join(lines))

    print(f"\nOutputs written to {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")
    print(f"  plots/: {len(list(PLOTS.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
