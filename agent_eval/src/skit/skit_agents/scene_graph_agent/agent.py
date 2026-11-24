"""Enhanced scene graph agent using OpenAI Agents SDK with spatial context."""

import os
import base64
import asyncio
from pathlib import Path
from typing import Optional
from PIL import Image
import io
from agents import Agent, Runner, RunConfig
from agents.extensions.models.litellm_model import LitellmModel
from skit.skit_agents.scene_graph_agent.scene_graph_schema import (
    SceneGraph,
)
from skit.vggt_camera_pose_estimator import VGGTCameraPoseEstimator
import logfire


class SceneGraphAgent:
    """Agent for generating scene graphs with spatial context from images."""

    def __init__(
        self,
        model_name: str = "gemini/gemini-2.5-pro",
        api_key: Optional[str] = None,
        scene_id: Optional[str] = None,
    ):
        """Initialize the enhanced scene graph agent.

        Args:
            model_name: LLM model to use (default: gemini-2.5-pro)
            api_key: API key for the model
            scene_id: Project identifier for tracking
            use_vggt_pose: Whether to use VGGT for camera pose estimation
            vggt_cache_dir: Cache directory for VGGT model
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.scene_id = scene_id

        self.agent = Agent(
            name="Scene Graph Generator",
            instructions=self._get_instructions(),
            model=LitellmModel(model=self.model_name, api_key=self.api_key),
            output_type=SceneGraph,
        )

    def _get_instructions(self) -> str:
        """Get detailed instructions for the scene graph generation agent."""
        instructions = """You are an expert in spatial reasoning and scene graph generation for infrastructure inspection.
        
        Your task is to create a comprehensive scene graph from inspection images with spatial context.

        For each image node:
        - Provide the exact image filename and path
        - Identify the central structural element or damage being documented
        - Provide a detailed description of what's visible
        - List all structural elements visible (beams, girders, deck, abutments, piers, bearings, etc.)
        - List any damage indicators (cracks, spalling, corrosion, deformation, etc.)
        - Create edges to other images that show:
        - The same element from different angles
        - Adjacent or connected structural elements
        - Related damage or conditions
        - Context relationships (wide shot to detail)

        For relative locations, be specific:
        - Use terms like "above", "below", "north of", "south of", "adjacent to"
        - Include approximate distances when possible
        - Reference structural landmarks

        For edges, describe the relationship clearly:
        - "Shows same beam from opposite angle"
        - "Adjacent span connection"
        - "Detail view of crack shown in wide shot"
        - "Continuation of girder line"

        Ensure the scene graph captures the complete spatial and semantic relationships between all images."""
        return instructions

    def encode_image(self, image_path: str, max_size: int = 512) -> str:
        """Encode image to base64, resizing if necessary to stay under size limits.

        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height) for the image

        Returns:
            Base64 encoded string of the (possibly resized) image
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Calculate resize ratio if image is too large
            width, height = img.size
            if max_size and (width > max_size or height > max_size):
                # Calculate scaling factor to fit within max_size
                scale = min(max_size / width, max_size / height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Resize the image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logfire.log(
                    "debug",
                    f"Resized {Path(image_path).name}: {width}x{height} -> {new_width}x{new_height}",
                )

            # Save to bytes
            img_byte_arr = io.BytesIO()
            # Use JPEG for photos (smaller file size)
            img_format = "JPEG" if img.mode == "RGB" else "PNG"
            img.save(
                img_byte_arr,
                format=img_format,
                quality=85 if img_format == "JPEG" else None,
            )
            img_byte_arr.seek(0)

            # Encode to base64
            return base64.b64encode(img_byte_arr.read()).decode("utf-8")

    def prepare_messages_batch(
        self,
        batch_files: list,
        # reference_nodes: ReferenceNodes,
        max_image_size: int = 512,
        batch_idx: int = 0,
        total_images: int = 0,
    ) -> list:
        """Prepare messages for a batch of images.

        Args:
            batch_files: List of Path objects for images in this batch
            reference_nodes: ReferenceNodes object for spatial context
            max_image_size: Maximum dimension for image resizing
            batch_idx: Starting index of this batch
            total_images: Total number of images across all batches
        """
        # Build content list
        content = []

        # Add context information
        context_text = f"PROJECT ID: {self.scene_id or 'unknown'}\n"

        context_text += f"BATCH INFO: Processing images {batch_idx + 1}-{batch_idx + len(batch_files)} of {total_images}\n"

        instruction_text = f"""Analyze these {len(batch_files)} inspection images and create scene graph nodes.

        {context_text}

        Generate EnhancedSceneNode entries for each image with all required fields populated.
        Focus on spatial relationships and structural elements."""

        content.append({"type": "input_text", "text": instruction_text})

        # Add each image with metadata
        for i, image_file in enumerate(batch_files):
            logfire.log(
                "debug", f"Adding image {i + 1}/{len(batch_files)}: {image_file.name}"
            )
            encoded_image = self.encode_image(str(image_file), max_size=max_image_size)

            # Always use JPEG mime type since we convert to JPEG in encode_image
            mime_type = "image/jpeg"

            # Add image
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded_image}",
                }
            )

            # Add image metadata
            content.append(
                {
                    "type": "input_text",
                    "text": f"Image {batch_idx + i + 1}: {image_file.name} (path: {image_file})",
                }
            )

        return [{"role": "user", "content": content}]

    async def _process_batch(
        self,
        batch_files: list,
        max_image_size: int,
        batch_idx: int,
        total_images: int,
        batch_num: int,
        disable_tracing: bool,
    ) -> SceneGraph:
        """Process a single batch of images asynchronously. Sends them to an agent to create a scene graph.

        Args:
            batch_files: List of Path objects for images in this batch
            max_image_size: Maximum dimension for image resizing
            batch_idx: Starting index of this batch
            total_images: Total number of images across all batches
            batch_num: Batch number for logging
            disable_tracing: Whether to disable tracing

        Returns:
            SceneGraph for this batch
        """
        try:
            logfire.info(f"Starting batch {batch_num}...")

            # Prepare messages for this batch
            messages = self.prepare_messages_batch(
                batch_files,
                max_image_size,
                batch_idx,
                total_images,
            )

            # Save debug info for first batch
            if batch_idx == 0:
                import json

                cleaned_messages = []
                for msg in messages:
                    cleaned_msg = {"role": msg["role"], "content": []}
                    for content_item in msg.get("content", []):
                        if content_item["type"] == "input_text":
                            cleaned_msg["content"].append(content_item)
                        elif content_item["type"] == "input_image":
                            cleaned_msg["content"].append(
                                {
                                    "type": "input_image",
                                    "image_url": "[IMAGE_DATA_REMOVED]",
                                }
                            )
                    cleaned_messages.append(cleaned_msg)

                debug_file = (
                    f"debug_scene_graph_messages_{self.scene_id or 'unknown'}.json"
                )
                with open(debug_file, "w") as f:
                    json.dump(cleaned_messages, f, indent=2)
                logfire.log("debug", f"Saved message structure to {debug_file}")

            config = RunConfig(tracing_disabled=disable_tracing)
            result = await Runner.run(self.agent, messages, run_config=config)

            # Get the structured output for this batch
            batch_graph = result.final_output

            logfire.info(f"Batch {batch_num} processing complete")
            return batch_graph

        except Exception as e:
            logfire.error(f"Error processing batch {batch_num}: {e}")
            # Return empty graph on error to not break the whole process
            return SceneGraph(
                nodes=[],
                scene_summary=f"Batch {batch_num} failed: {str(e)}",
            )

    def prepare_messages(
        self,
        folder_path: str,
        query: Optional[str] = None,
        max_image_size: int = 512,
        max_images: Optional[int] = None,
    ) -> list:
        """Prepare messages with images and reference context.

        Args:
            folder_path: Path to folder containing images
            query: Optional query or context for analysis
            max_image_size: Maximum dimension for image resizing (default: 512)
            max_images: Maximum number of images to process (None for all)
        """
        folder = Path(folder_path)

        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        image_files = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        )

        if not image_files:
            raise ValueError(f"No images found in {folder_path}")

        # Limit number of images if specified
        if max_images and len(image_files) > max_images:
            logfire.info(
                f"Limiting to {max_images} images out of {len(image_files)} found"
            )
            image_files = image_files[:max_images]

        logfire.info(
            f"Processing {len(image_files)} images (max size: {max_image_size}x{max_image_size})"
        )

        # Build content list
        content = []

        # Add context information
        context_text = f"PROJECT ID: {self.scene_id or 'unknown'}\n"
        if query:
            context_text += f"ANALYSIS CONTEXT: {query}\n"

        # Iterate through the reference nodes (now a ReferenceNodes object)

        # Add instruction with context
        instruction_text = f"""Analyze these {len(image_files)} inspection images and create a comprehensive scene graph.

        {context_text}

        Generate a complete SceneGraph with all required fields populated."""

        content.append({"type": "input_text", "text": instruction_text})

        # Add each image with metadata
        for i, image_file in enumerate(image_files):
            logfire.log(
                "debug", f"Adding image {i + 1}/{len(image_files)}: {image_file.name}"
            )
            encoded_image = self.encode_image(str(image_file), max_size=max_image_size)

            # Always use JPEG mime type since we convert to JPEG in encode_image
            mime_type = "image/jpeg"

            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded_image}",
                }
            )

            content.append(
                {
                    "type": "input_text",
                    "text": f"Image {i + 1} - Filename: {image_file.name}, Path: {str(image_file)}",
                }
            )

        return [{"role": "user", "content": content}]

    async def generate_scene_graph(
        self,
        folder_path: str,
        disable_tracing: bool = True,
        max_image_size: int = 512,
        batch_size: int = 16,
        max_concurrent_batches: Optional[int] = None,
    ) -> SceneGraph:
        """Generate an enhanced scene graph from a folder of images.

        Args:
            folder_path: Path to folder containing images
            disable_tracing: Whether to disable tracing for this run
            max_image_size: Maximum dimension for image resizing (default: 512)
            batch_size: Number of images to process at a time (default: 16)
            max_concurrent_batches: Maximum number of batches to process in parallel (None for all)
            output_dir: Optional output directory for COLMAP reconstruction
            generate_colmap: Whether to generate COLMAP 3D reconstruction
            colmap_use_ba: Whether to use bundle adjustment in COLMAP

        Returns:
            EnhancedSceneGraph with all nodes and relationships
        """
        # Always auto-compute reference nodes from images
        logfire.info("Auto-computing reference nodes from images...")

        # Get all image files
        folder = Path(folder_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        all_image_files = sorted(
            [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        )

        if not all_image_files:
            raise ValueError(f"No images found in {folder_path}")

        total_images = len(all_image_files)
        logfire.info(f"Total images to process: {total_images}")

        # Create batch tasks for parallel processing
        batch_tasks = []
        batch_info = []

        for batch_idx in range(0, total_images, batch_size):
            batch_end = min(batch_idx + batch_size, total_images)
            batch_files = all_image_files[batch_idx:batch_end]
            batch_num = batch_idx // batch_size + 1

            batch_info.append(
                {
                    "num": batch_num,
                    "start": batch_idx + 1,
                    "end": batch_end,
                    "files": batch_files,
                }
            )

            # Create async task for this batch
            task = self._process_batch(
                batch_files,
                max_image_size,
                batch_idx,
                total_images,
                batch_num,
                disable_tracing,
            )
            batch_tasks.append(task)

        logfire.info(f"Processing {len(batch_tasks)} batches in parallel...")
        if max_concurrent_batches:
            logfire.info(f"Maximum concurrent batches: {max_concurrent_batches}")
        else:
            logfire.info(f"Processing all {len(batch_tasks)} batches concurrently")

        for info in batch_info:
            logfire.log(
                "debug",
                f"Batch {info['num']}: images {info['start']}-{info['end']} ({len(info['files'])} images)",
            )

        # Execute batches with optional concurrency limit
        if max_concurrent_batches and max_concurrent_batches < len(batch_tasks):
            # Process batches with limited concurrency using a semaphore
            semaphore = asyncio.Semaphore(max_concurrent_batches)

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            batch_results = await asyncio.gather(
                *[run_with_semaphore(task) for task in batch_tasks]
            )
        else:
            # Execute all batches in parallel
            batch_results = await asyncio.gather(*batch_tasks)

        # Collect all nodes from all batches
        all_nodes = []
        for batch_num, batch_graph in enumerate(batch_results, 1):
            if batch_graph and batch_graph.nodes:
                all_nodes.extend(batch_graph.nodes)
                logfire.info(
                    f"Batch {batch_num} complete: {len(batch_graph.nodes)} nodes processed"
                )

        for index, node in enumerate(all_nodes):
            all_nodes[index].image_name = os.path.basename(node.image_name)

        # Create final scene graph with all nodes
        final_scene_graph = SceneGraph(
            nodes=all_nodes,
        )

        return final_scene_graph

    async def generate_colmap_from_folder(
        self,
        folder_path: str,
        output_dir: str,
        use_ba: bool = False,
        max_reproj_error: float = 8.0,
        shared_camera: bool = True,
        camera_type: str = "SIMPLE_PINHOLE",
        query_frame_num: int = 40,
        max_query_pts: int = 4096,
        fine_tracking: bool = True,
    ) -> bool:
        """Generate COLMAP 3D reconstruction from a folder of images.

        This can be called independently after scene graph generation
        or as part of the scene graph generation process.

        Args:
            folder_path: Path to folder containing images
            output_dir: Directory to save COLMAP reconstruction (will create sparse/ subfolder)
            use_ba: Whether to use bundle adjustment for refinement
            max_reproj_error: Maximum reprojection error threshold
            shared_camera: Whether all images share the same camera parameters
            camera_type: COLMAP camera model type
            query_frame_num: Number of query frames for tracking
            max_query_pts: Maximum query points for tracking
            fine_tracking: Whether to use fine tracking (slower but more accurate)

        Returns:
            True if reconstruction succeeded, False otherwise
        """

        logfire.info("Generating COLMAP 3D reconstruction...")

        try:
            vggt_estimator = VGGTCameraPoseEstimator(
                cache_dir=".vggt_cache", enable_cache=False
            )
            # Get all image files from folder
            folder = Path(folder_path)
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            image_files = sorted(
                [
                    f
                    for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
            )

            if not image_files:
                logfire.error(f"No images found in {folder_path}")
                return False

            # Get all image paths
            image_paths = [str(f) for f in image_files]
            logfire.info(
                f"Processing {len(image_paths)} images for COLMAP reconstruction"
            )

            # Generate COLMAP reconstruction
            success = vggt_estimator.generate_colmap_reconstruction(
                image_paths=image_paths,
                output_dir=output_dir,
                use_ba=use_ba,
                max_reproj_error=max_reproj_error,
                shared_camera=shared_camera,
                camera_type=camera_type,
                query_frame_num=query_frame_num,
                max_query_pts=max_query_pts,
                fine_tracking=fine_tracking,
            )

            if success:
                logfire.info(f"COLMAP reconstruction saved to: {output_dir}/sparse")
            else:
                logfire.error("COLMAP reconstruction failed")
                raise RuntimeError("COLMAP reconstruction failed")
            return success

        except Exception as e:
            logfire.error(f"Error generating COLMAP reconstruction: {e}")
            import traceback

            traceback.print_exc()
            return False

    def generate_scene_graph_sync(
        self,
        folder_path: str,
        disable_tracing: bool = True,
        max_image_size: int = 512,
        batch_size: int = 16,
        max_concurrent_batches: Optional[int] = None,
    ) -> SceneGraph:
        """Synchronous wrapper for scene graph generation."""
        return asyncio.run(
            self.generate_scene_graph(
                folder_path,
                disable_tracing,
                max_image_size,
                batch_size,
                max_concurrent_batches,
            )
        )
