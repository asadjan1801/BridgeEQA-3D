"""
DataFrame building utilities for experiment results.

Provides functions to transform experiment results into pandas DataFrames
for analysis and visualization.
"""

from typing import Any, Dict, List

import pandas as pd


def build_results_dataframe(
    agent_results: Dict[str, List[Dict[str, Any]]],
    agents_to_include: List[str] | None = None,
    filter_no_condition_rating: bool = True
) -> pd.DataFrame:
    """
    Build a wide-format DataFrame from agent results.

    By default, filters out questions that don't have condition ratings since they
    shouldn't be included in condition rating metric calculations.

    The DataFrame will have columns:
    - scene_name: Name of the scene directory
    - question: The inspection question
    - reference_answer: Ground truth answer
    - reference_condition_rating: Ground truth condition rating (1-9 scale)
    - reference_images: List of ground truth reference image filenames
    - <agent_name>_answer: Agent's answer (for each agent)
    - <agent_name>_condition_rating: Agent's condition rating (for each agent)
    - <agent_name>_reference_images: Images referenced by agent (for each agent)

    Args:
        agent_results: Dictionary mapping agent names to result records
        agents_to_include: Optional list of agent names to filter
                          If None, includes all agents in agent_results
        filter_no_condition_rating: If True (default), exclude questions without condition ratings
                                   to prevent them from affecting metric calculations

    Returns:
        pandas DataFrame with aggregated results in wide format

    Example:
        >>> results = load_experiments_from_db("experiments.db")
        >>> df = build_results_dataframe(results, agents_to_include=["agent1", "agent2"])
        >>> print(df.columns)
        ['scene_name', 'question', 'reference_answer', 'reference_condition_rating',
         'reference_images', 'agent1_answer', 'agent1_condition_rating',
         'agent1_reference_images', 'agent2_answer', ...]
    """
    # Filter agents if specified
    if agents_to_include is not None:
        agent_results = {
            name: results
            for name, results in agent_results.items()
            if name in agents_to_include
        }

    # Collect all unique (scene_name, question) combinations with their reference condition ratings
    scene_questions_with_ratings = {}
    for results_list in agent_results.values():
        for result in results_list:
            key = (result['scene_dir'], result['question'])
            if key not in scene_questions_with_ratings:
                scene_questions_with_ratings[key] = result.get('reference_condition_rating')

    # Filter out questions without condition ratings if requested
    if filter_no_condition_rating:
        original_count = len(scene_questions_with_ratings)
        scene_questions_with_ratings = {
            k: v for k, v in scene_questions_with_ratings.items()
            if v is not None
        }
        filtered_count = original_count - len(scene_questions_with_ratings)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} questions without condition ratings")

    scene_questions = set(scene_questions_with_ratings.keys())

    # Build rows for the DataFrame
    rows = []
    for scene_name, question in sorted(scene_questions):
        row = {
            'scene_name': scene_name,
            'question': question,
        }

        # Find reference data from any agent (should be same across all agents)
        for results_list in agent_results.values():
            for result in results_list:
                if result['scene_dir'] == scene_name and result['question'] == question:
                    row['reference_answer'] = result['reference_answer']
                    row['reference_condition_rating'] = result['reference_condition_rating']
                    row['reference_images'] = result.get('reference_images', [])
                    break
            if 'reference_answer' in row:
                break

        # Add agent-specific columns
        for agent_name, results_list in agent_results.items():
            # Find this agent's result for this scene/question
            agent_result = None
            for result in results_list:
                if result['scene_dir'] == scene_name and result['question'] == question:
                    agent_result = result
                    break

            if agent_result:
                row[f"{agent_name}_answer"] = agent_result['agent_answer']
                row[f"{agent_name}_reference_images"] = agent_result.get('agent_reference_images', [])

                # Handle condition rating: set to None if reference is None
                if row.get('reference_condition_rating') is None:
                    row[f"{agent_name}_condition_rating"] = None
                else:
                    row[f"{agent_name}_condition_rating"] = agent_result['agent_condition_rating']
            else:
                # Agent didn't process this scene/question
                row[f"{agent_name}_answer"] = None
                row[f"{agent_name}_condition_rating"] = None
                row[f"{agent_name}_reference_images"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nBuilt DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df
