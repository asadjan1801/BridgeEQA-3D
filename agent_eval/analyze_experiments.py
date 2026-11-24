"""
Analyze agent experiments using SQLite databases.

This script loads experiment results from the ExperimentTracker database
and provides comprehensive analysis including:
- Condition rating accuracy metrics
- Answer correctness evaluation with LLM judge
- VLM-based evaluation with ground truth images
- Visualization of agent performance
- Comparison across multiple models/databases

Usage:
    python analyze_experiments_from_db.py
"""

import os
import asyncio
from pathlib import Path
from typing import Dict

import pandas as pd

from skit.evaluation import (
    load_experiments_from_db,
    build_results_dataframe,
    postprocess_embodied_agent_images,
    analyze_agent_responses,
    analyze_answer_correctness,
    analyze_image_relevance,
    evaluate_answers_with_text_judge,
    evaluate_answers_with_vlm_judge,
)


async def main(
    db_path: Path | str,
    agents_to_include: list[str] | None = None,
    run_llm_judge: bool = False,
    run_vlm_judge: bool = False,
    vlm_sample_size: int | None = 100,
    vlm_random_seed: int = 42,
    output_prefix: str = "",
    agent_name_mapping: Dict[str, str] | None = None,
    scene_dir_override: Path | str | None = None
) -> pd.DataFrame:
    """
    Main analysis pipeline using database as source.

    Args:
        db_path: Path to experiments.db file
        agents_to_include: Optional list of agent names to filter
        run_llm_judge: Whether to evaluate answers with text-based LLM judge (slow)
        run_vlm_judge: Whether to evaluate answers with VLM judge using ground truth images (expensive)
        vlm_sample_size: Number of random samples for VLM evaluation per agent (default: 100, None = all)
        vlm_random_seed: Random seed for VLM sampling reproducibility (default: 42)
        output_prefix: Prefix for output files (e.g., "grok-4-fast_")
        agent_name_mapping: Optional mapping of agent names to display names
        scene_dir_override: Optional path to scene directories (if different from db parent)

    Returns:
        DataFrame with aggregated results
    """
    # Load experiments from database
    agent_results = load_experiments_from_db(db_path)

    # Postprocess embodied agent images to convert node references
    # Determine base scene directory
    if scene_dir_override:
        base_scene_dir = Path(scene_dir_override)
    else:
        # Default: assume scenes are in pdf_processed_vermont_pages_greater_than_20
        db_path = Path(db_path)
        base_scene_dir = Path("pdf_processed_vermont_pages_greater_than_20")

    agent_results = postprocess_embodied_agent_images(agent_results, base_scene_dir)

    # Build DataFrame (automatically filters out questions without condition ratings)
    df = build_results_dataframe(agent_results, agents_to_include, filter_no_condition_rating=True)

    # Save DataFrame
    csv_filename = f"{output_prefix}aggregated_agent_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved DataFrame to {csv_filename}")
    print(f"DataFrame contains {len(df)} rows with condition ratings")

    # Run analysis
    if len(df) > 0:
        analyze_agent_responses(df, output_prefix, agent_name_mapping)
    else:
        print("No questions with condition ratings found. Skipping condition rating analysis.")

    # Optional: LLM judge evaluation
    if run_llm_judge:
        print("\n" + "="*80)
        print("Running Text-based LLM Judge Evaluation (this may take a while)...")
        print("="*80)
        df = await evaluate_answers_with_text_judge(
            df,
            output_csv=f"{output_prefix}aggregated_agent_results_with_correctness.csv"
        )

        # Analyze correctness scores
        analyze_answer_correctness(df, output_prefix, agent_name_mapping)

    # Optional: VLM judge evaluation with ground truth images
    if run_vlm_judge:
        print("\n" + "="*80)
        if vlm_sample_size:
            print(f"Running VLM Judge Evaluation with Random Sampling (n={vlm_sample_size} per agent)...")
        else:
            print("Running VLM Judge Evaluation on Full Dataset (expensive!)...")
        print("="*80)
        df = await evaluate_answers_with_vlm_judge(
            df,
            base_scene_dir=base_scene_dir,
            output_csv=f"{output_prefix}aggregated_agent_results_with_vlm_correctness.csv",
            sample_size=vlm_sample_size,
            random_seed=vlm_random_seed,
            max_concurrent=10
        )

        # Analyze VLM correctness scores
        analyze_answer_correctness(df, output_prefix, agent_name_mapping)

    # Analyze image relevance
    analyze_image_relevance(df, output_prefix, agent_name_mapping)

    return df


async def run_all_analyses():
    """
    Run analyses for all models in a single event loop.

    This prevents event loop mismatch issues with litellm's background logging.
    """
    data_dir = "pdf_processed_vermont_pages_greater_than_20_eval_"
    model_names = ["gemini-2.5-flash", "grok-4-fast", "gemini-2.5-flash-lite"]

    for model_name in model_names:
        # Configuration
        DB_PATH = Path(f"{data_dir}{model_name}_cached_output/experiments.db")

        # Specify which agents to include in the DataFrame
        # Set to None to include all agents
        AGENTS_TO_INCLUDE = ["embodied_no_tools_both", "embodied_no_tools_images",
                            "embodied_no_tools_sg_context", "embodied_sg_oracle_both",
                            "embodied_sg_oracle_sg_context"]

        Agent_Name_Mapping = {
            "embodied_no_tools_both": "VLM w/ Images + SG",
            "embodied_no_tools_images": "VLM w/ Images Only",
            "embodied_no_tools_sg_context": "VLM w/ SG Only",
            "embodied_sg_oracle_both": "Embodied Agent w/ Images + SG",
            "embodied_sg_oracle_sg_context": "Embodied Agent w/ SG Only",
        }

        # Whether to run LLM judge evaluation (can be slow)
        RUN_LLM_JUDGE = True

        # Whether to run VLM judge evaluation with ground truth images (expensive!)
        RUN_VLM_JUDGE = True

        # VLM sampling configuration (uses random sampling for cost efficiency)
        VLM_SAMPLE_SIZE = 200  # Number of samples per agent (None = evaluate all)
        VLM_RANDOM_SEED = 42   # For reproducibility

        output_dir = "graph_outputs_larger"
        os.makedirs(output_dir, exist_ok=True)

        # Output prefix for files
        OUTPUT_PREFIX = f"{output_dir}/{model_name}_"

        # Run analysis
        df = await main(
            db_path=DB_PATH,
            agents_to_include=AGENTS_TO_INCLUDE,
            run_llm_judge=RUN_LLM_JUDGE,
            run_vlm_judge=RUN_VLM_JUDGE,
            vlm_sample_size=VLM_SAMPLE_SIZE,
            vlm_random_seed=VLM_RANDOM_SEED,
            output_prefix=OUTPUT_PREFIX,
            agent_name_mapping=Agent_Name_Mapping
        )

        print("\n" + "="*80)
        print(f"Analysis complete for {model_name}!")
        print("="*80)


if __name__ == "__main__":
    # Run all analyses in a single event loop to prevent litellm event loop issues
    asyncio.run(run_all_analyses())
