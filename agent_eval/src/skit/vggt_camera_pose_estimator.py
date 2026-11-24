import os
import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import blake3
import pickle
from PIL import Image
import io
import shutil
import random
import copy
import torch.nn.functional as F
import trimesh
import pycolmap
import logfire

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import (
    load_and_preprocess_images,
    load_and_preprocess_images_square,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from .vggt_output_types import CameraPoseResult, VGGTFullPredictions
from .vgg_utils import predictions_to_glb


class VGGTCameraPoseEstimator:
    """
    A self-contained class for estimating camera poses from images using VGGT.

    Example usage:
        estimator = VGGTCameraPoseEstimator()
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        extrinsics, intrinsics = estimator.get_camera_poses(image_paths)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the VGGT model for camera pose estimation.

        Args:
            model_path: Optional path to a local model checkpoint. If None, downloads from HuggingFace.
            device: Device to run inference on ('cuda' or 'cpu'). If None, auto-detects.
            cache_dir: Directory to store cache files. If None, uses '.vggt_cache' in current directory.
            enable_cache: Whether to enable caching of results based on image content.
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize cache
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or ".vggt_cache"
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_cache_from_disk()

        logfire.info(f"Initializing VGGT on device: {self.device}")
        if self.enable_cache:
            logfire.info(f"Cache enabled, storing in: {self.cache_dir}")

        # Initialize model
        self.model = VGGT()

        # Load model weights
        if model_path and os.path.exists(model_path):
            logfire.info(f"Loading model from local path: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logfire.info("Loading model from HuggingFace hub...")
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        logfire.info("Model loaded successfully!")

    def _compute_images_hash(self, image_paths: List[str]) -> str:
        """
        Compute a hash based on the actual content of images.

        Args:
            image_paths: List of paths to image files

        Returns:
            Blake3 hash string representing the combined image content
        """
        hasher = blake3.blake3()

        for path in sorted(image_paths):  # Sort to ensure consistent ordering
            # Read image and get its raw bytes
            with Image.open(path) as img:
                # Convert to RGB to ensure consistent format
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Get image bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()
                # Update hash with image content and path
                hasher.update(img_bytes)
                hasher.update(path.encode("utf-8"))

        return hasher.hexdigest()

    def _get_cache_key(self, image_paths: List[str], method: str = "poses") -> str:
        """
        Generate a cache key based on image content and method.

        Args:
            image_paths: List of paths to image files
            method: Type of prediction ('poses' or 'full')

        Returns:
            Cache key string
        """
        images_hash = self._compute_images_hash(image_paths)
        return f"{method}_{images_hash}"

    def _load_cache_from_disk(self) -> None:
        """Load cache from disk if it exists."""
        cache_file = os.path.join(self.cache_dir, "vggt_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logfire.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logfire.log("warning", f"Could not load cache: {e}")
                self.cache = {}

    def _save_cache_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.enable_cache:
            return

        cache_file = os.path.join(self.cache_dir, "vggt_cache.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logfire.log("warning", f"Could not save cache: {e}")

    def get_camera_poses(self, image_paths: List[Union[str, Path]]) -> CameraPoseResult:
        """
        Estimate camera poses for a list of images.

        Args:
            image_paths: List of paths to image files

        Returns:
            CameraPoseResult containing:
            - extrinsic_matrices: (N, 4, 4) array of camera extrinsic matrices
            - intrinsic_matrices: (N, 3, 3) array of camera intrinsic matrices
            - image_paths: List of image paths corresponding to the matrices
            where N is the number of images
        """
        if not image_paths:
            raise ValueError("No image paths provided")

        # Convert paths to strings and check they exist
        image_paths = [str(p) for p in image_paths]
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")

        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(image_paths, "poses")
            if cache_key in self.cache:
                self.cache_hits += 1
                logfire.info(
                    f"Cache hit! Returning cached camera poses for {len(image_paths)} images"
                )
                cached_result = self.cache[cache_key]
                return CameraPoseResult(
                    extrinsic_matrices=cached_result["extrinsic"],
                    intrinsic_matrices=cached_result["intrinsic"],
                    image_paths=cached_result["image_paths"],
                )
            else:
                self.cache_misses += 1

        logfire.info(f"Processing {len(image_paths)} images...")

        # Load and preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)
        logfire.info(f"Preprocessed images shape: {images.shape}")

        # Run inference
        with torch.no_grad():
            # Use appropriate dtype for GPU
            if self.device == "cuda":
                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    predictions = self.model(images)
            else:
                predictions = self.model(images)

        # Convert pose encoding to camera matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:],  # (H, W)
        )

        # Convert to numpy and remove batch dimension if present
        if isinstance(extrinsic, torch.Tensor):
            extrinsic = extrinsic.cpu().numpy()
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.cpu().numpy()

        # Remove batch dimension if present (shape is [1, N, 3, 4] -> [N, 3, 4])
        if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
            extrinsic = extrinsic.squeeze(0)
        if intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        logfire.info("Camera poses extracted:")
        logfire.info(f"  - Extrinsic matrices shape: {extrinsic.shape}")
        logfire.info(f"  - Intrinsic matrices shape: {intrinsic.shape}")

        # Cache the result if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(image_paths, "poses")
            self.cache[cache_key] = {
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "image_paths": image_paths,
            }
            self._save_cache_to_disk()

        return CameraPoseResult(
            extrinsic_matrices=extrinsic,
            intrinsic_matrices=intrinsic,
            image_paths=image_paths,
        )

    def get_full_predictions(
        self, image_paths: List[Union[str, Path]]
    ) -> VGGTFullPredictions:
        """
        Get full predictions including camera poses, depth maps, and point clouds.

        Args:
            image_paths: List of paths to image files

        Returns:
            VGGTFullPredictions containing:
            - extrinsic_matrices: Camera extrinsic matrices
            - intrinsic_matrices: Camera intrinsic matrices
            - depth: Predicted depth maps
            - depth_conf: Depth confidence maps
            - world_points: 3D world points
            - world_points_conf: World points confidence
            - image_paths: List of image paths corresponding to the predictions
        """
        if not image_paths:
            raise ValueError("No image paths provided")

        # Convert paths to strings
        image_paths = [str(p) for p in image_paths]

        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(image_paths, "full")
            if cache_key in self.cache:
                self.cache_hits += 1
                logfire.info(
                    f"Cache hit! Returning cached full predictions for {len(image_paths)} images"
                )
                cached_dict = self.cache[cache_key]
                return VGGTFullPredictions(
                    extrinsic_matrices=cached_dict["extrinsic"],
                    intrinsic_matrices=cached_dict["intrinsic"],
                    depth=cached_dict["depth"],
                    depth_conf=cached_dict["depth_conf"],
                    world_points=cached_dict["world_points"],
                    world_points_conf=cached_dict["world_points_conf"],
                    pose_enc=cached_dict.get("pose_enc"),
                    images=cached_dict.get("images"),
                    image_paths=cached_dict["image_paths"],
                )
            else:
                self.cache_misses += 1

        logfire.info(f"Processing {len(image_paths)} images for full predictions...")

        # Load and preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)

        # Run inference
        with torch.no_grad():
            if self.device == "cuda":
                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    predictions = self.model(images)
            else:
                predictions = self.model(images)

        # Convert pose encoding to camera matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )

        # Add camera matrices to predictions
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert all tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()
                # Remove batch dimension if present
                if predictions[key].ndim > 2 and predictions[key].shape[0] == 1:
                    predictions[key] = predictions[key].squeeze(0)

        # Remove pose_enc_list if present (not needed in output)
        if "pose_enc_list" in predictions:
            predictions.pop("pose_enc_list")

        # Add image paths to predictions
        predictions["image_paths"] = image_paths

        # Cache the result if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(image_paths, "full")
            self.cache[cache_key] = predictions
            self._save_cache_to_disk()

        return VGGTFullPredictions(
            extrinsic_matrices=predictions["extrinsic"],
            intrinsic_matrices=predictions["intrinsic"],
            depth=predictions["depth"],
            depth_conf=predictions["depth_conf"],
            world_points=predictions["world_points"],
            world_points_conf=predictions["world_points_conf"],
            pose_enc=predictions.get("pose_enc"),
            images=predictions.get("images"),
            image_paths=predictions["image_paths"],
        )

    def save_camera_poses(
        self, image_paths: List[Union[str, Path]], output_path: Union[str, Path]
    ) -> None:
        """
        Process images and save camera poses to a numpy file.

        Args:
            image_paths: List of paths to image files
            output_path: Path to save the .npz file with camera matrices
        """
        result = self.get_camera_poses(image_paths)

        # Save to file
        np.savez(
            output_path,
            extrinsic=result.extrinsic_matrices,
            intrinsic=result.intrinsic_matrices,
            image_paths=result.image_paths,
        )

        logfire.info(f"Camera poses saved to: {output_path}")

    def clear_cache(self) -> None:
        """Clear the in-memory and disk cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if self.enable_cache:
            cache_file = os.path.join(self.cache_dir, "vggt_cache.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logfire.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "enabled": self.enable_cache,
            "entries": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "cache_dir": self.cache_dir if self.enable_cache else None,
        }

        if self.enable_cache:
            cache_file = os.path.join(self.cache_dir, "vggt_cache.pkl")
            if os.path.exists(cache_file):
                stats["cache_size_mb"] = os.path.getsize(cache_file) / (1024 * 1024)
            else:
                stats["cache_size_mb"] = 0

        return stats

    def generate_colmap_reconstruction(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        use_ba: bool = False,
        max_reproj_error: float = 16.0,
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        query_frame_num: int = 40,
        max_query_pts: int = 4096,
        fine_tracking: bool = True,
        conf_thres_value: float = 1.5,
        vis_thresh: float = 0.2,
        seed: int = 42,
        max_points_for_colmap: int = 3000000,
    ) -> bool:
        """Generate COLMAP 3D reconstruction using VGGT predictions.

        This directly generates a full 3D reconstruction including sparse point cloud
        and camera poses in COLMAP format using the already-loaded VGGT model.

        Args:
            image_paths: List of paths to image files
            output_dir: Directory to save COLMAP reconstruction (will create sparse/ subfolder)
            use_ba: Whether to use bundle adjustment for refinement
            max_reproj_error: Maximum reprojection error threshold
            shared_camera: Whether all images share the same camera parameters
            camera_type: COLMAP camera model type
            query_frame_num: Number of query frames for tracking
            max_query_pts: Maximum query points for tracking
            fine_tracking: Whether to use fine tracking (slower but more accurate)
            conf_thres_value: Confidence threshold for depth filtering (without BA)
            vis_thresh: Visibility threshold for tracks
            seed: Random seed for reproducibility

        Returns:
            True if reconstruction succeeded, False otherwise
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        output_dir = Path(output_dir)
        scene_dir = output_dir
        images_dir = scene_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to the expected location if needed
        image_path_list = []
        base_image_path_list = []
        for img_path in image_paths:
            img_path = Path(img_path)
            dest_path = images_dir / img_path.name
            if not dest_path.exists() or dest_path.resolve() != img_path.resolve():
                shutil.copy2(img_path, dest_path)
            image_path_list.append(str(dest_path))
            base_image_path_list.append(img_path.name)

        logfire.info(f"Running VGGT reconstruction for {len(image_paths)} images...")
        logfire.info(f"Output will be saved to: {scene_dir / 'sparse'}")
        logfire.info(f"Device: {self.device}")

        # Check CUDA availability if using GPU
        if self.device == "cuda":
            if not torch.cuda.is_available():
                logfire.error("CUDA requested but not available!")
                return False
            logfire.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logfire.info(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB total"
            )
            torch.cuda.empty_cache()  # Clear any cached memory

        try:
            logfire.info("Loading and processing images for VGGT Inference")
            # Load images at higher resolution for COLMAP
            vggt_fixed_resolution = 518
            img_load_resolution = 1024

            # Load and preprocess images
            logfire.info(
                f"Loading {len(image_path_list)} images at resolution {img_load_resolution}x{img_load_resolution}"
            )
            try:
                images, original_coords = load_and_preprocess_images_square(
                    image_path_list, img_load_resolution
                )
                images = images.to(self.device)
                original_coords = original_coords.to(self.device)
                logfire.info(f"Images loaded successfully. Shape: {images.shape}")
            except Exception as e:
                logfire.error(f"Failed to load images: {e}")
                if "out of memory" in str(e).lower():
                    logfire.info(
                        "Trying with lower resolution (518x518) due to memory constraints..."
                    )
                    torch.cuda.empty_cache()
                    img_load_resolution = 518
                    images, original_coords = load_and_preprocess_images_square(
                        image_path_list, img_load_resolution
                    )
                    images = images.to(self.device)
                    original_coords = original_coords.to(self.device)
                else:
                    raise

            # Run VGGT inference at 518x518
            logfire.info(
                f"Resizing images to VGGT resolution ({vggt_fixed_resolution}x{vggt_fixed_resolution})"
            )
            images_vggt = F.interpolate(
                images,
                size=(vggt_fixed_resolution, vggt_fixed_resolution),
                mode="bilinear",
                align_corners=False,
            )

            logfire.info("Starting VGGT model inference...")
            with torch.no_grad():
                if self.device == "cuda":
                    dtype = (
                        torch.bfloat16
                        if torch.cuda.get_device_capability()[0] >= 8
                        else torch.float16
                    )
                    logfire.info(f"Using dtype: {dtype}")
                    try:
                        with torch.amp.autocast("cuda", dtype=dtype):
                            images_batch = images_vggt[None]  # add batch dimension
                            logfire.info("Running aggregator...")
                            aggregated_tokens_list, ps_idx = self.model.aggregator(
                                images_batch
                            )
                            logfire.info("Running camera head...")
                            pose_enc = self.model.camera_head(aggregated_tokens_list)[
                                -1
                            ]
                            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                                pose_enc, images_vggt.shape[-2:]
                            )
                            logfire.info("Running depth head...")
                            depth_map, depth_conf = self.model.depth_head(
                                aggregated_tokens_list, images_batch, ps_idx
                            )
                    except torch.cuda.OutOfMemoryError as e:
                        logfire.error(f"CUDA out of memory: {e}")
                        torch.cuda.empty_cache()
                        raise RuntimeError(
                            f"GPU out of memory processing {len(image_paths)} images. Try reducing the number of images or using CPU."
                        )
                else:
                    images_batch = images_vggt[None]
                    logfire.info("Running aggregator (CPU)...")
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
                    logfire.info("Running camera head (CPU)...")
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc, images_vggt.shape[-2:]
                    )
                    logfire.info("Running depth head (CPU)...")
                    depth_map, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )

            logfire.info(f"finished VGGT inference, starting COLMAP data conversion...")
            # Convert to numpy
            extrinsic = extrinsic.squeeze(0).cpu().numpy()
            intrinsic = intrinsic.squeeze(0).cpu().numpy()
            depth_map = depth_map.squeeze(0).cpu().numpy()
            depth_conf = depth_conf.squeeze(0).cpu().numpy()

            # Unproject to 3D points
            points_3d = unproject_depth_map_to_point_map(
                depth_map, extrinsic, intrinsic
            )

            if use_ba:
                image_size = np.array(images.shape[-2:])
                scale = img_load_resolution / vggt_fixed_resolution

                with (
                    torch.amp.autocast("cuda", dtype=dtype)
                    if self.device == "cuda"
                    else torch.no_grad()
                ):
                    # Predict tracks for bundle adjustment
                    pred_tracks, pred_vis_scores, _, points_3d, points_rgb = (
                        predict_tracks(
                            images,
                            conf=depth_conf,
                            points_3d=points_3d,
                            masks=None,
                            max_query_pts=max_query_pts,
                            query_frame_num=query_frame_num,
                            keypoint_extractor="aliked+sp",
                            fine_tracking=fine_tracking,
                        )
                    )
                    torch.cuda.empty_cache()

                # Rescale intrinsics from 512 to 1024
                intrinsic[:, :2, :] *= scale
                track_mask = pred_vis_scores > vis_thresh

                # Convert to COLMAP format with tracks
                reconstruction, _ = batch_np_matrix_to_pycolmap(
                    points_3d,
                    extrinsic,
                    intrinsic,
                    pred_tracks,
                    image_size,
                    masks=track_mask,
                    max_reproj_error=max_reproj_error,
                    shared_camera=shared_camera,
                    camera_type=camera_type,
                    points_rgb=points_rgb,
                )

                if reconstruction is None:
                    logfire.error("No reconstruction could be built with BA")
                    return False

                # Bundle Adjustment
                ba_options = pycolmap.BundleAdjustmentOptions()
                pycolmap.bundle_adjustment(reconstruction, ba_options)

                reconstruction_resolution = img_load_resolution
            else:
                image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
                num_frames, height, width, _ = points_3d.shape

                # Get RGB values for points
                points_rgb = F.interpolate(
                    images,
                    size=(vggt_fixed_resolution, vggt_fixed_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
                points_rgb = points_rgb.transpose(0, 2, 3, 1)

                # Create pixel coordinate grid
                points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

                # Apply confidence threshold
                conf_mask = depth_conf >= conf_thres_value
                conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

                points_3d_filtered = points_3d[conf_mask]
                points_xyf_filtered = points_xyf[conf_mask]
                points_rgb_filtered = points_rgb[conf_mask]

                # Convert to COLMAP format without tracks
                reconstruction = batch_np_matrix_to_pycolmap_wo_track(
                    points_3d_filtered,
                    points_xyf_filtered,
                    points_rgb_filtered,
                    extrinsic,
                    intrinsic,
                    image_size,
                    shared_camera=shared_camera,
                    camera_type="PINHOLE" if not use_ba else camera_type,
                )

                reconstruction_resolution = vggt_fixed_resolution

            # Rename and rescale cameras to original resolution
            reconstruction = self._rename_and_rescale_colmap(
                reconstruction,
                base_image_path_list,
                original_coords.cpu().numpy(),
                img_size=reconstruction_resolution,
                shift_point2d_to_original_res=True,
                shared_camera=shared_camera,
            )

            # Save reconstruction
            sparse_dir = scene_dir / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            reconstruction.write(str(sparse_dir))

            points_3d_vis = points_3d_filtered
            points_rgb_vis = points_rgb_filtered

            trimesh.PointCloud(points_3d_vis, colors=points_rgb_vis).export(
                str(sparse_dir / "points.ply")
            )

            # Generate GLB file for 3D visualization using predictions_to_glb
            try:
                glb_path = scene_dir / "scene.glb"

                # Get the original unmodified points for GLB generation
                # We need the full resolution points, not the filtered ones
                points_3d_for_glb = unproject_depth_map_to_point_map(
                    depth_map, extrinsic, intrinsic
                )

                # Prepare images at VGGT resolution (518x518) for GLB generation
                # The images_vggt are already at the correct resolution
                if isinstance(images_vggt, torch.Tensor):
                    images_np = images_vggt.cpu().numpy()
                else:
                    images_np = images_vggt

                # Ensure images are in (S, H, W, 3) format, not (S, 3, H, W)
                if images_np.ndim == 4 and images_np.shape[1] == 3:
                    # Transpose from NCHW to NHWC
                    images_np = np.transpose(images_np, (0, 2, 3, 1))

                # Ensure depth_conf has the right shape (S, H, W, 1) -> (S, H, W)
                if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
                    depth_conf_squeezed = depth_conf.squeeze(-1)
                else:
                    depth_conf_squeezed = depth_conf

                # Prepare predictions dict for predictions_to_glb
                # The function expects world_points in shape (S, H, W, 3)
                predictions_for_glb = {
                    "world_points": points_3d_for_glb,  # Shape: (S, H, W, 3)
                    "world_points_conf": depth_conf_squeezed,  # Shape: (S, H, W)
                    "depth_conf": depth_conf_squeezed,  # Add this for vgg_utils.py compatibility
                    "images": images_np,  # Shape: (S, H, W, 3) at 518x518
                    "extrinsic": extrinsic,  # Shape: (S, 3, 4)
                }

                # Also add world_points_from_depth as fallback (as in the example)
                predictions_for_glb["world_points_from_depth"] = points_3d_for_glb

                # Generate GLB scene
                glb_scene = predictions_to_glb(
                    predictions_for_glb,
                    conf_thres=50.0,  # Use 50th percentile threshold
                    show_cam=True,
                    prediction_mode="Depthmap",  # Match the example
                    max_points_for_colmap=max_points_for_colmap,
                )

                # Export to GLB file
                glb_scene.export(str(glb_path))
                logfire.info(f"GLB scene saved to: {glb_path}")
            except Exception as e:
                logfire.log("warning", f"Could not generate GLB: {e}")
                import traceback

                traceback.print_exc()

            logfire.info("COLMAP reconstruction completed successfully!")
            logfire.info(f"Files saved to: {sparse_dir}")
            logfire.info(
                f"Point cloud visualization available at: {sparse_dir / 'points.ply'}"
            )
            return True

        except Exception as e:
            logfire.info(f"Error generating COLMAP reconstruction: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _rename_and_rescale_colmap(
        self,
        reconstruction,
        image_paths,
        original_coords,
        img_size,
        shift_point2d_to_original_res=False,
        shared_camera=False,
    ):
        """Rename images and rescale camera parameters to original resolution."""
        rescale_camera = True

        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid - 1]

            if rescale_camera:
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = original_coords[pyimageid - 1, -2:]
                resize_ratio = max(real_image_size) / img_size
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp

                pycamera.params = pred_params
                pycamera.width = int(real_image_size[0])
                pycamera.height = int(real_image_size[1])

            if shift_point2d_to_original_res:
                top_left = original_coords[pyimageid - 1, :2]
                resize_ratio = max(original_coords[pyimageid - 1, -2:]) / img_size

                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                rescale_camera = False

        return reconstruction

    def __del__(self):
        """Clean up GPU memory and save cache when object is destroyed."""
        if hasattr(self, "enable_cache") and self.enable_cache:
            self._save_cache_to_disk()

        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
