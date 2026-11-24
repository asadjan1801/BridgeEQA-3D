"""
SKIT Evaluation Package

This package provides tools for evaluating agent performance on bridge inspection tasks.

Components:
- caches: SQLite-based caching for evaluation results
- judges: LLM and VLM judges for answer correctness evaluation
- metrics: Similarity and accuracy metrics
- loaders: Database loading utilities for experiment results
- dataframes: DataFrame building and manipulation
- analysis: Visualization and statistical analysis functions
- batch_evaluation: Async batch evaluation with rate limiting
- postprocessing: Node reference and image conversion utilities
"""

from .caches import TextCorrectnessCache, VlmCorrectnessCache
from .judges import (
    create_text_judge_agent,
    create_vlm_judge_agent,
    evaluate_answer_correctness,
    evaluate_answer_correctness_async,
    evaluate_vlm_answer_correctness_async,
    CORRECTNESS_PROMPT,
    VLM_CORRECTNESS_PROMPT,
)
from .metrics import calculate_jaccard_similarity
from .loaders import load_experiments_from_db, load_experiments_from_multiple_dbs
from .dataframes import build_results_dataframe
from .analysis import (
    get_display_name,
    analyze_agent_responses,
    analyze_answer_correctness,
    analyze_image_relevance,
)
from .batch_evaluation import (
    evaluate_answers_with_text_judge,
    evaluate_answers_with_vlm_judge,
)
from .postprocessing import (
    convert_node_references_to_image_paths,
    postprocess_embodied_agent_images,
    prepare_ground_truth_images,
)

__all__ = [
    # Caches
    "TextCorrectnessCache",
    "VlmCorrectnessCache",
    # Judges
    "create_text_judge_agent",
    "create_vlm_judge_agent",
    "evaluate_answer_correctness",
    "evaluate_answer_correctness_async",
    "evaluate_vlm_answer_correctness_async",
    "CORRECTNESS_PROMPT",
    "VLM_CORRECTNESS_PROMPT",
    # Metrics
    "calculate_jaccard_similarity",
    # Loaders
    "load_experiments_from_db",
    "load_experiments_from_multiple_dbs",
    # DataFrames
    "build_results_dataframe",
    # Analysis
    "get_display_name",
    "analyze_agent_responses",
    "analyze_answer_correctness",
    "analyze_image_relevance",
    # Batch Evaluation
    "evaluate_answers_with_text_judge",
    "evaluate_answers_with_vlm_judge",
    # Postprocessing
    "convert_node_references_to_image_paths",
    "postprocess_embodied_agent_images",
    "prepare_ground_truth_images",
]
