"""
Batch evaluation utilities with async processing and rate limiting.

Provides functions to evaluate multiple agent answers in parallel with
proper rate limiting using asyncio semaphores.
"""

import asyncio
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stats

from .judges import evaluate_answer_correctness_async, evaluate_vlm_answer_correctness_async


async def evaluate_answers_batch_async(
    df: pd.DataFrame,
    agent_answer_columns: List[str],
    max_concurrent: int = 10
) -> pd.DataFrame:
    """
    Evaluate all agent answers in parallel with rate limiting (text-based LLM judge).

    Args:
        df: DataFrame with agent results
        agent_answer_columns: List of column names containing agent answers
        max_concurrent: Maximum number of concurrent LLM requests (default: 5)

    Returns:
        Updated DataFrame with correctness score columns
    """
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"\n=== Starting Async Evaluation (max {max_concurrent} concurrent requests) ===\n")

    for col in agent_answer_columns:
        agent_name = col.replace('_answer', '')
        print(f"Evaluating correctness scores for {agent_name}...")

        # Build list of tasks for this agent
        tasks = []
        task_indices = []

        for idx, row in df.iterrows():
            answer = row[col]
            ground_truth = row['reference_answer']

            if pd.isna(answer) or pd.isna(ground_truth):
                # Skip None/NaN values
                continue
            else:
                # Create async task
                task = evaluate_answer_correctness_async(
                    answer=answer,
                    ground_truth=ground_truth,
                    semaphore=semaphore
                )
                tasks.append(task)
                task_indices.append(idx)

        # Run all tasks concurrently (semaphore limits actual concurrency)
        print(f"  Processing {len(tasks)} evaluations...")
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Build correctness scores dict (using index as key to handle non-sequential indices)
        correctness_scores_dict = {}
        for idx, score in zip(task_indices, scores):
            if isinstance(score, Exception):
                print(f"  Error evaluating row {idx}: {score}")
                correctness_scores_dict[idx] = 0.0
            else:
                correctness_scores_dict[idx] = score

        # Add to DataFrame (align by index)
        df[f"{agent_name}_correctness_score"] = df.index.map(correctness_scores_dict)
        print(f"  Completed evaluation for {agent_name}\n")

    return df


async def evaluate_answers_with_vlm_batch_async(
    df: pd.DataFrame,
    agent_answer_columns: List[str],
    base_scene_dir: Path,
    max_concurrent: int = 3,  # was 10 — Gemini 503s under burstier load
    sample_size: int | None = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate agent answers with VLM judge (with ground truth images) using random sampling.

    Validates that all referenced images exist before evaluation. If a sampled row
    references missing images, it will be skipped and another sample will be drawn
    to maintain the target sample size.

    Args:
        df: DataFrame with agent results
        agent_answer_columns: List of column names containing agent answers
        base_scene_dir: Base directory containing scene folders
        max_concurrent: Maximum number of concurrent VLM requests (default: 3)
        sample_size: Number of random samples to evaluate per agent (default: None = all)
        random_seed: Random seed for reproducible sampling (default: 42)

    Returns:
        Updated DataFrame with VLM correctness score columns (NaN for non-sampled rows)
    """
    # Create semaphore for rate limiting (lower for VLM)
    semaphore = asyncio.Semaphore(max_concurrent)

    if sample_size:
        print(f"\n=== Starting Async VLM Evaluation with Random Sampling ===")
        print(f"Sample size: {sample_size} per agent (seed: {random_seed})")
        print(f"Max concurrent requests: {max_concurrent}\n")
    else:
        print(f"\n=== Starting Async VLM Evaluation (max {max_concurrent} concurrent requests) ===\n")

    for col in agent_answer_columns:
        agent_name = col.replace('_answer', '')
        # print(f"Evaluating VLM correctness scores for {agent_name}...")

        # Filter valid rows (non-NaN answers and questions)
        valid_mask = df[col].notna() & df['question'].notna()
        valid_indices = df[valid_mask].index.tolist()

        if not valid_indices:
            print(f"  No valid data for {agent_name}, skipping.")
            continue

        # Helper function to check if all images exist for a given row
        def _check_images_exist(idx: int, agent_col: str) -> bool:
            """
            Check if all reference images AND agent-selected images exist.

            Args:
                idx: DataFrame row index
                agent_col: Agent answer column name (to derive image column)

            Returns:
                True if all required images exist, False otherwise
            """
            row = df.loc[idx]
            reference_images = row.get('reference_images', [])
            scene_dir = row['scene_name']

            if not reference_images:
                return False

            scene_path = base_scene_dir / scene_dir / "images"

            # Check reference images exist
            for img_filename in reference_images:
                img_name = Path(img_filename).name
                image_path = scene_path / img_name

                if not image_path.exists():
                    return False

            # Also check agent's selected images exist (since we show them to VLM)
            agent_images_col = agent_col.replace('_answer', '_reference_images')
            agent_images = row.get(agent_images_col, []) if agent_images_col in df.columns else []

            if agent_images:
                for img_filename in agent_images:
                    img_name = Path(img_filename).name
                    image_path = scene_path / img_name

                    if not image_path.exists():
                        return False

            return True

        # Sample if requested, with image validation and resampling
        if sample_size and sample_size < len(valid_indices):
            random.seed(random_seed)
            sampled_indices = []
            available_indices = valid_indices.copy()
            random.shuffle(available_indices)

            skipped_missing_images = 0

            for idx in available_indices:
                if len(sampled_indices) >= sample_size:
                    break

                # Check if all referenced images exist (both reference and agent-selected)
                if _check_images_exist(idx, col):
                    sampled_indices.append(idx)
                else:
                    skipped_missing_images += 1

            if skipped_missing_images > 0:
                print(f"  Skipped {skipped_missing_images} samples due to missing images (reference or agent-selected)")

            print(f"  Sampled {len(sampled_indices)} valid samples (target: {sample_size})")

            if len(sampled_indices) < sample_size:
                print(f"  Warning: Only found {len(sampled_indices)} samples with valid images (target was {sample_size})")
        else:
            # For full evaluation, filter out rows with missing images
            sampled_indices = []
            skipped_missing_images = 0

            for idx in valid_indices:
                if _check_images_exist(idx, col):
                    sampled_indices.append(idx)
                else:
                    skipped_missing_images += 1

            if skipped_missing_images > 0:
                print(f"  Skipped {skipped_missing_images} samples due to missing images (reference or agent-selected)")

            print(f"  Evaluating {len(sampled_indices)} responses with valid images")

        if not sampled_indices:
            print(f"  No valid samples with existing images for {agent_name}, skipping.")
            continue

        # Build list of tasks for sampled indices
        tasks = []
        task_indices = []

        for idx in sampled_indices:
            row = df.loc[idx]
            answer = row[col]
            question = row['question']
            reference_images = row.get('reference_images', [])
            scene_dir = row['scene_name']

            # Get the agent's selected images (same column name pattern as answer)
            agent_images_col = col.replace('_answer', '_reference_images')
            agent_images = row.get(agent_images_col, []) if agent_images_col in df.columns else []

            # Create async task
            task = evaluate_vlm_answer_correctness_async(
                answer=answer,
                question=question,
                reference_images=reference_images,
                scene_dir=scene_dir,
                base_scene_dir=base_scene_dir,
                semaphore=semaphore,
                agent_reference_images=agent_images
            )
            tasks.append(task)
            task_indices.append(idx)

        # Run all tasks concurrently (semaphore limits actual concurrency)
        print(f"  Processing {len(tasks)} evaluations...")
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Build correctness scores dict (using index as key to handle non-sequential indices)
        vlm_correctness_scores_dict = {}
        for idx, score in zip(task_indices, scores):
            if isinstance(score, Exception):
                print(f"  Error evaluating row {idx}: {score}")
                vlm_correctness_scores_dict[idx] = 0.0
            else:
                vlm_correctness_scores_dict[idx] = score

        # Add to DataFrame (align by index)
        df[f"{agent_name}_vlm_correctness_score"] = df.index.map(vlm_correctness_scores_dict)
        print(f"  Completed VLM evaluation for {agent_name}\n")

    return df


async def evaluate_answers_with_text_judge(
    df: pd.DataFrame,
    output_csv: str = "aggregated_agent_results_with_correctness.csv",
    max_concurrent: int = 3,  # was 10 — Gemini 503s under burstier load
) -> pd.DataFrame:
    """
    Evaluate correctness of each agent's answer using text-based LLM judge.

    Uses async processing with semaphore rate limiting for better performance.

    Args:
        df: DataFrame with agent results
        output_csv: Path to save DataFrame with correctness scores
        max_concurrent: Maximum number of concurrent LLM requests (default: 10)

    Returns:
        Updated DataFrame with correctness score columns
    """
    # Get agent answer columns
    agent_answer_columns = [
        col for col in df.columns
        if col.endswith('_answer') and col != 'reference_answer'
    ]

    if not agent_answer_columns:
        print("No agent answer columns found.")
        return df

    # Run async evaluation
    df = await evaluate_answers_batch_async(df, agent_answer_columns, max_concurrent)

    # Save updated DataFrame with correctness scores
    df.to_csv(output_csv)
    print(f"Saved updated DataFrame with correctness scores to {output_csv}")
    return df


async def evaluate_answers_with_vlm_judge(
    df: pd.DataFrame,
    base_scene_dir: Path,
    output_csv: str = "aggregated_agent_results_with_vlm_correctness.csv",
    max_concurrent: int = 3,  # was 10 — Gemini 503s under burstier load
    sample_size: int | None = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate correctness of each agent's answer using VLM judge with ground truth images.

    Uses async processing with semaphore rate limiting for better performance.
    Supports random sampling for cost-effective evaluation.

    Args:
        df: DataFrame with agent results
        base_scene_dir: Base directory containing scene folders
        output_csv: Path to save DataFrame with VLM correctness scores
        max_concurrent: Maximum number of concurrent VLM requests (default: 3)
        sample_size: Number of random samples per agent (default: None = all)
        random_seed: Random seed for reproducible sampling (default: 42)

    Returns:
        Updated DataFrame with VLM correctness score columns
    """
    # Get agent answer columns
    agent_answer_columns = [
        col for col in df.columns
        if col.endswith('_answer') and col != 'reference_answer'
    ]

    if not agent_answer_columns:
        print("No agent answer columns found.")
        return df

    # Run async VLM evaluation with sampling
    df = await evaluate_answers_with_vlm_batch_async(
        df,
        agent_answer_columns,
        base_scene_dir,
        max_concurrent,
        sample_size=sample_size,
        random_seed=random_seed
    )

    # Compute and print sampling statistics if sampling was used
    if sample_size:
        print("\n" + "="*80)
        print("VLM SAMPLING STATISTICS (95% Confidence Intervals)")
        print("="*80)

        for col in agent_answer_columns:
            agent_name = col.replace('_answer', '')
            score_col = f"{agent_name}_vlm_correctness_score"

            if score_col in df.columns:
                scores = df[score_col].dropna()

                if len(scores) > 0:
                    mean = scores.mean()
                    std = scores.std()
                    n = len(scores)

                    # 95% confidence interval using t-distribution
                    ci = stats.t.interval(
                        0.95,
                        df=n-1,
                        loc=mean,
                        scale=std/np.sqrt(n)
                    )

                    print(f"\n{agent_name}:")
                    print(f"  Sample Size: {n}")
                    print(f"  Mean Score: {mean:.3f}")
                    print(f"  Std Dev: {std:.3f}")
                    print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
                    print(f"  Margin of Error: ±{(ci[1] - mean):.3f}")

        print("="*80 + "\n")

    # Save updated DataFrame with VLM correctness scores
    df.to_csv(output_csv)
    print(f"Saved updated DataFrame with VLM correctness scores to {output_csv}")
    return df
