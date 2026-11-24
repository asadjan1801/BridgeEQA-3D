"""
Postprocessing utilities for agent evaluation results.

Provides functions to convert node references to image paths and prepare
images for VLM evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List


def convert_node_references_to_image_paths(
    reference_images: List[str] | None,
    scene_graph_path: Path
) -> List[str] | None:
    """
    Convert node references (e.g., "node 21", "Node 11") to actual image filenames.

    The agent may return references in various formats:
    - "node 21" or "Node 21" (node index)
    - "Abutment 1.png" (descriptive name that doesn't match actual filename)
    - Actual image hash filenames (already correct)

    This function:
    1. Checks if the reference is already a valid image filename (hash format)
    2. Tries to parse node indices and convert to image paths
    3. Tries exact caption matching using scene graph and image_reference_mappings.json
    4. Tries fuzzy caption matching (substring matching)
    5. Falls back to trying to match against actual image names

    Args:
        reference_images: List of image references from agent output
        scene_graph_path: Path to the scene_graph.json file

    Returns:
        List of actual image filenames (just the basename, not full path)
        Returns None if reference_images is None
        Returns original list if scene graph not found or conversion fails
    """
    if not reference_images:
        return None

    if not scene_graph_path.exists():
        print(f"Warning: Scene graph not found at {scene_graph_path}")
        return reference_images

    # Load scene graph
    try:
        from skit.skit_agents.scene_graph_schema import SceneGraph
        scene_graph = SceneGraph.model_validate_json(open(scene_graph_path).read())
    except Exception as e:
        print(f"Warning: Failed to load scene graph: {e}")
        return reference_images

    converted_images = []
    scene_dir = scene_graph_path.parent
    image_dir = scene_dir / "images"

    # Create mapping of node index to image filename (basename only)
    # Also create caption-to-filename mapping from scene graph central_focus
    node_to_filename = {}
    caption_to_filename = {}

    for idx, node in enumerate(scene_graph.nodes):
        # Extract just the filename from the full path
        filename = Path(node.image_name).name
        node_to_filename[idx] = filename

        # Map central_focus to filename (lowercase, strip for matching)
        if hasattr(node, 'central_focus') and node.central_focus:
            caption_to_filename[node.central_focus.lower().strip()] = filename

    # Also load caption-to-filename mapping from image_reference_mappings.json
    # This provides additional mapping from PDF captions to image filenames
    mapping_file = scene_dir / "image_reference_mappings.json"
    if mapping_file.exists():
        try:
            import json
            mappings = json.load(open(mapping_file))
            for entry in mappings:
                caption = entry.get("reference_images")
                filename = entry.get("image_name")
                if caption and filename:
                    # Skip corrupted filenames (valid hash filenames are 32 hex chars + .png = 36 total)
                    # Only include if file actually exists OR has valid hash format
                    is_valid_hash = (len(filename) == 36 and filename.endswith('.png') and
                                    all(c in '0123456789abcdef' for c in filename[:-4]))
                    file_exists = (image_dir / filename).exists()

                    if not (is_valid_hash or file_exists):
                        # Skip this corrupted mapping entry
                        continue

                    # Normalize caption for matching (lowercase, strip whitespace)
                    if isinstance(caption, str):
                        caption_to_filename[caption.lower().strip()] = filename
                    elif isinstance(caption, list):
                        for cap in caption:
                            caption_to_filename[str(cap).lower().strip()] = filename
        except Exception as e:
            print(f"Warning: Failed to load image_reference_mappings.json: {e}")

    for ref in reference_images:
        ref_str = str(ref).strip()

        # Check if it's already a valid image hash filename (32-char hex + .png)
        if len(ref_str) == 36 and ref_str.endswith('.png') and all(c in '0123456789abcdef' for c in ref_str[:-4]):
            converted_images.append(ref_str)
            continue

        # Try to parse as "node N" or "Node N"
        ref_lower = ref_str.lower()
        if ref_lower.startswith('node '):
            try:
                node_idx = int(ref_str.split()[-1])
                if node_idx in node_to_filename:
                    converted_images.append(node_to_filename[node_idx])
                    continue
            except (ValueError, IndexError):
                pass

        # Try to parse as just a number (node index)
        try:
            node_idx = int(ref_str)
            if node_idx in node_to_filename:
                converted_images.append(node_to_filename[node_idx])
                continue
        except ValueError:
            pass

        # Try exact caption match first
        if ref_lower in caption_to_filename:
            matched_filename = caption_to_filename[ref_lower]
            # Validate the match - check if file exists
            if (image_dir / matched_filename).exists():
                converted_images.append(matched_filename)
                continue
            else:
                # Fallback: The filename in mappings might be corrupted
                # Don't use this corrupted filename - fall through to fuzzy matching
                pass

        # Try fuzzy caption matching - check if the reference contains or is contained in any caption
        # This handles cases like:
        #   - Mapping has "Page 16, Figure: Abutment 1 (south) joint"
        #   - Agent returns "Abutment 1 (south) joint"
        #   - Scene graph has "Abutment 1 (south) joint"
        best_match = None
        best_match_score = 0
        for caption, filename in caption_to_filename.items():
            # Check if reference is contained in caption or vice versa
            if ref_lower in caption or caption in ref_lower:
                # Prefer longer matches (more specific)
                score = min(len(ref_lower), len(caption))
                if score > best_match_score:
                    best_match = filename
                    best_match_score = score

        if best_match:
            converted_images.append(best_match)
            continue

        # Check if it matches any actual image file in the directory
        if image_dir.exists():
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    if image_path.name == ref_str or image_path.stem in ref_str:
                        converted_images.append(image_path.name)
                        break

    return converted_images if converted_images else None


def postprocess_embodied_agent_images(
    agent_results: Dict[str, List[Dict[str, Any]]],
    base_scene_dir: Path
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Postprocess embodied agent results to convert node references to image filenames.

    This function:
    1. Identifies agents that use scene graph node references (embodied agents)
    2. For each experiment, loads the scene graph and converts node refs to filenames
    3. Updates agent_reference_images in place

    Args:
        agent_results: Dictionary mapping agent names to lists of experiment results
        base_scene_dir: Base directory where scene directories are located

    Returns:
        Updated agent_results dictionary with converted image references
    """
    print("\n" + "="*80)
    print("POSTPROCESSING: Converting node references to image filenames")
    print("="*80)

    # Identify embodied agents (those that use scene graph tools)
    embodied_agent_patterns = ['sg_oracle']

    for agent_name, experiments in agent_results.items():
        # Check if this is an embodied agent
        is_embodied = any(pattern in agent_name.lower() for pattern in embodied_agent_patterns)

        if not is_embodied:
            continue

        print(f"\nProcessing embodied agent: {agent_name}")
        converted_count = 0
        failed_count = 0
        already_correct = 0

        for exp in experiments:
            agent_images = exp.get('agent_reference_images', [])

            # Handle None, empty list, or non-list types (like float/NaN)
            if not agent_images or not isinstance(agent_images, list):
                continue

            # Check if already in correct format (all are hash filenames)
            all_correct = all(
                len(str(img)) == 36 and str(img).endswith('.png') and
                all(c in '0123456789abcdef' for c in str(img)[:-4])
                for img in agent_images
            )

            if all_correct:
                already_correct += 1
                continue

            # Try to convert
            scene_dir = base_scene_dir / exp['scene_dir']
            scene_graph_path = scene_dir / 'scene_graph.json'

            converted = convert_node_references_to_image_paths(
                agent_images,
                scene_graph_path
            )

            if converted and converted != agent_images:
                exp['agent_reference_images'] = converted
                converted_count += 1
            else:
                failed_count += 1

        print(f"  Converted: {converted_count} experiments")
        print(f"  Already correct: {already_correct} experiments")
        print(f"  Failed to convert: {failed_count} experiments")

    print("="*80 + "\n")
    return agent_results


def prepare_ground_truth_images(
    reference_images: List[str],
    scene_dir: str,
    base_scene_dir: Path
) -> List[dict]:
    """
    Load ground truth images and convert to VLM-compatible format.

    This function loads image files from the scene directory and converts them
    to the format expected by the OpenAI Agents SDK (presigned URLs).

    Args:
        reference_images: List of image filenames (e.g., ['abc123.png', 'def456.png'])
        scene_dir: Scene directory name (e.g., 'BridgeInspRpt-BARNET-00010')
        base_scene_dir: Base directory containing scene folders

    Returns:
        List of dicts with {"type": "input_image", "image_url": url}
        Empty list if images cannot be loaded
    """
    if not reference_images:
        return []

    from skit.files import CloudFileStorage

    cfs = CloudFileStorage()
    image_dicts = []

    scene_path = base_scene_dir / scene_dir / "images"

    for img_filename in reference_images:
        # Handle both formats: just filename or full path
        img_name = Path(img_filename).name

        # Try to find the image in the scene directory
        image_path = scene_path / img_name

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            # Get presigned URL
            file_response = cfs.get_download_url(image_path)
            image_dicts.append({
                "type": "input_image",
                "image_url": file_response.download_url
            })
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            continue

    return image_dicts
