"""
Metrics for evaluating agent performance.

Provides similarity metrics, accuracy metrics, and error metrics for comparing
agent outputs with ground truth.
"""

from pathlib import Path
from typing import List


def calculate_jaccard_similarity(agent_images: List[str], reference_images: List[str]) -> float:
    """
    Calculate Jaccard similarity (Intersection over Union) between agent and reference images.

    The Jaccard similarity coefficient is defined as:
        J(A, B) = |A ∩ B| / |A ∪ B|

    This metric is used to evaluate how well the agent's image selection matches
    the ground truth reference images.

    Args:
        agent_images: List of image paths/filenames selected by the agent
        reference_images: List of ground truth reference image paths/filenames

    Returns:
        Float between 0.0 and 1.0 representing overlap ratio (intersection/union)
        Returns 0.0 if both lists are empty or no overlap exists

    Example:
        >>> agent_imgs = ["image1.png", "image2.png", "image3.png"]
        >>> ref_imgs = ["image2.png", "image3.png", "image4.png"]
        >>> calculate_jaccard_similarity(agent_imgs, ref_imgs)
        0.5  # 2 images in common out of 4 unique images
    """
    # Extract just the filename (not the full path) for comparison
    agent_set = set(Path(p).name for p in agent_images)
    reference_set = set(Path(p).name for p in reference_images)

    # Calculate intersection and union
    intersection = agent_set.intersection(reference_set)
    union = agent_set.union(reference_set)

    # Handle empty case
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)
