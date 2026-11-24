"""Run LLM + VLM judge analysis under gemini-3.1-flash-lite judge for three models:
gemini-3.1-flash-lite (newly evaluated), SC fine-tune, MC fine-tune.

Sequential in a single event loop (see analyze_experiments.run_all_analyses for the
rationale — litellm's background logging breaks across loop boundaries).

Outputs land in (fresh) graph_outputs_* dirs; old gemini-2.5-flash-lite-judged
counterparts were preserved at *_judge25fl/.

Run: cd agent_eval && uv run python run_analysis_judge31fl.py
"""

import asyncio
import os
from pathlib import Path

from analyze_experiments import main


AGENTS_TO_INCLUDE = [
    "embodied_no_tools_both",
    "embodied_no_tools_images",
    "embodied_no_tools_sg_context",
    "embodied_sg_oracle_both",
    "embodied_sg_oracle_sg_context",
]

AGENT_NAME_MAPPING = {
    "embodied_no_tools_both": "VLM w/ Images + SG",
    "embodied_no_tools_images": "VLM w/ Images Only",
    "embodied_no_tools_sg_context": "VLM w/ SG Only",
    "embodied_sg_oracle_both": "Embodied Agent w/ Images + SG",
    "embodied_sg_oracle_sg_context": "Embodied Agent w/ SG Only",
}

SCENE_DIR = Path(os.environ.get("BRIDGE_ROOT", "/home/projects")) / "pdf_processed_vermont_pages_greater_than_20"

JOBS = [
    # In the original run this first job finished before a resume and was then
    # commented out; it is enabled here so the script covers all three Table 2 columns.
    {
        "label": "gemini-3.1-flash-lite (eval model)",
        "db_path": Path(
            "pdf_processed_vermont_pages_greater_than_20_eval_"
            "gemini-3.1-flash-lite_cached_output/experiments.db"
        ),
        "output_dir": "graph_outputs_gemini31fl",
        "output_prefix_basename": "gemini-3.1-flash-lite_",
    },
    {
        "label": "SC fine-tune",
        "db_path": Path(
            "pdf_processed_vermont_pages_greater_than_20_eval_"
            "gemma4-bridge_cached_output_v3mixed_BACKUP/experiments.db"
        ),
        "output_dir": "graph_outputs_gemma4_bridge_finetuned_SC",
        "output_prefix_basename": "gemma4-bridge-sc_",
    },
    {
        "label": "MC fine-tune",
        "db_path": Path(
            "pdf_processed_vermont_pages_greater_than_20_eval_"
            "gemma4-bridge_cached_output/experiments.db"
        ),
        "output_dir": "graph_outputs_gemma4_bridge_finetuned",
        "output_prefix_basename": "gemma4-bridge_",
    },
]


async def run_all() -> None:
    for job in JOBS:
        print("\n" + "#" * 80)
        print(f"### {job['label']}")
        print(f"### DB:  {job['db_path']}")
        print(f"### Out: {job['output_dir']}/")
        print("#" * 80)

        assert job["db_path"].exists(), f"missing DB: {job['db_path']}"
        os.makedirs(job["output_dir"], exist_ok=True)

        await main(
            db_path=job["db_path"],
            agents_to_include=AGENTS_TO_INCLUDE,
            run_llm_judge=True,
            run_vlm_judge=True,
            vlm_sample_size=200,
            vlm_random_seed=42,
            output_prefix=f"{job['output_dir']}/{job['output_prefix_basename']}",
            agent_name_mapping=AGENT_NAME_MAPPING,
            scene_dir_override=SCENE_DIR,
        )

        print(f"\n[done] {job['label']}")

    print("\n" + "=" * 80)
    print("All three analyses complete under gemini-3.1-flash-lite judge.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_all())
