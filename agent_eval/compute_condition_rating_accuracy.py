"""Compute Exact-Match and Within-±1 condition-rating accuracy for the three
models (gemini-3.1-flash-lite eval, SC fine-tune, MC fine-tune) from the
aggregated CSVs produced by analyze_experiments.main. Writes one summary CSV
per model and prints a side-by-side table.

The accuracy_comparison.png plots already exist in each graph_outputs dir but the
numeric values are not persisted to CSV by the upstream pipeline.

Inputs are the graph_outputs_*/ directories written by run_analysis_judge31fl.py.

Run: cd agent_eval && uv run python compute_condition_rating_accuracy.py
"""

from pathlib import Path
import pandas as pd

AGENT_NAME_MAPPING = {
    "embodied_no_tools_images": "VLM w/ Images Only",
    "embodied_no_tools_sg_context": "VLM w/ SG Only",
    "embodied_no_tools_both": "VLM w/ Images + SG",
    "embodied_sg_oracle_sg_context": "Embodied Agent w/ SG Only",
    "embodied_sg_oracle_both": "Embodied Agent w/ Images + SG",
}
AGENT_ORDER = list(AGENT_NAME_MAPPING.keys())

MODELS = [
    ("Gemini-3.1 Flash-Lite",
     "graph_outputs_gemini31fl/gemini-3.1-flash-lite_aggregated_agent_results.csv"),
    ("Ours SC",
     "graph_outputs_gemma4_bridge_finetuned_SC/gemma4-bridge-sc_aggregated_agent_results.csv"),
    ("Ours MC",
     "graph_outputs_gemma4_bridge_finetuned/gemma4-bridge_aggregated_agent_results.csv"),
]


def compute_one(label: str, csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ref = df["reference_condition_rating"]
    rows = []
    for agent in AGENT_ORDER:
        col = f"{agent}_condition_rating"
        if col not in df.columns:
            continue
        pred = df[col]
        mask = ref.notna() & pred.notna()
        n = int(mask.sum())
        if n == 0:
            continue
        r = ref[mask]
        p = pred[mask]
        exact = int((p == r).sum())
        within1 = int((abs(p - r) <= 1).sum())
        rows.append({
            "Agent": AGENT_NAME_MAPPING[agent],
            "N": n,
            "Exact Match %": 100.0 * exact / n,
            "Within ±1 %": 100.0 * within1 / n,
        })
    out = pd.DataFrame(rows)
    return out


def render_combined(per_model: dict[str, pd.DataFrame]) -> None:
    agents = [AGENT_NAME_MAPPING[a] for a in AGENT_ORDER]

    def fmt_table(metric_col: str, title: str) -> str:
        head = f"### {title}\n\n"
        head += "| Agent | " + " | ".join(per_model.keys()) + " |\n"
        head += "|---|" + "---|" * len(per_model) + "\n"
        lines = []
        for agent in agents:
            cells = []
            for model_label, df in per_model.items():
                v = df.loc[df["Agent"] == agent, metric_col]
                cells.append(f"{v.iloc[0]:.1f}" if len(v) else "—")
            lines.append(f"| {agent} | " + " | ".join(cells) + " |")
        # mean
        cells = []
        for model_label, df in per_model.items():
            cells.append(f"**{df[metric_col].mean():.1f}**")
        lines.append("| **Mean** | " + " | ".join(cells) + " |")
        return head + "\n".join(lines) + "\n"

    print(fmt_table("Exact Match %",
                    "Condition Rating Accuracy — Exact Match (%)"))
    print()
    print(fmt_table("Within ±1 %",
                    "Condition Rating Accuracy — Within ±1 (%)"))


def main() -> None:
    per_model: dict[str, pd.DataFrame] = {}
    for label, csv_path in MODELS:
        p = Path(csv_path)
        assert p.exists(), f"missing CSV: {p}"
        df = compute_one(label, str(p))
        out_csv = p.with_name(p.stem + "_condition_rating_accuracy.csv")
        df.to_csv(out_csv, index=False)
        print(f"[wrote] {out_csv}")
        per_model[label] = df

    print()
    render_combined(per_model)


if __name__ == "__main__":
    main()
