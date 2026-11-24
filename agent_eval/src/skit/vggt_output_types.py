"""Pydantic models for VGGT and DINOv3 output data structures."""

from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field



class CameraPoseResult(BaseModel):
    """Result from camera pose estimation."""
    
    extrinsic_matrices: np.ndarray = Field(
        ..., 
        description="Camera extrinsic matrices of shape (N, 3, 4) or (N, 4, 4)"
    )
    intrinsic_matrices: np.ndarray = Field(
        ..., 
        description="Camera intrinsic matrices of shape (N, 3, 3)"
    )
    image_paths: List[str] = Field(
        ..., 
        description="List of image paths corresponding to the camera matrices"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def __len__(self) -> int:
        """Return the number of camera poses."""
        return len(self.image_paths)
    
    def get_translations(self) -> np.ndarray:
        """Extract translation vectors from extrinsic matrices."""
        if self.extrinsic_matrices.shape[-1] == 4 and self.extrinsic_matrices.shape[-2] >= 3:
            return self.extrinsic_matrices[:, :3, 3]
        else:
            raise ValueError(f"Cannot extract translations from shape: {self.extrinsic_matrices.shape}")


class VGGTFullPredictions(BaseModel):
    """Complete VGGT model predictions including poses, depth, and point clouds."""
    
    # Camera parameters
    extrinsic_matrices: np.ndarray = Field(
        ..., 
        description="Camera extrinsic matrices of shape (N, 3, 4) or (N, 4, 4)"
    )
    intrinsic_matrices: np.ndarray = Field(
        ..., 
        description="Camera intrinsic matrices of shape (N, 3, 3)"
    )
    
    # Depth information
    depth: np.ndarray = Field(
        ..., 
        description="Predicted depth maps"
    )
    depth_conf: np.ndarray = Field(
        ..., 
        description="Depth confidence maps"
    )
    
    # 3D point information
    world_points: np.ndarray = Field(
        ..., 
        description="3D world points"
    )
    world_points_conf: np.ndarray = Field(
        ..., 
        description="World points confidence"
    )
    
    # Additional model outputs
    pose_enc: Optional[np.ndarray] = Field(
        None, 
        description="Pose encoding from the model"
    )
    images: Optional[np.ndarray] = Field(
        None, 
        description="Preprocessed input images"
    )
    
    # Metadata
    image_paths: List[str] = Field(
        ..., 
        description="List of image paths corresponding to all predictions"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def __len__(self) -> int:
        """Return the number of predictions."""
        return len(self.image_paths)
    
    def get_camera_poses(self) -> CameraPoseResult:
        """Extract camera pose information as a separate result."""
        return CameraPoseResult(
            extrinsic_matrices=self.extrinsic_matrices,
            intrinsic_matrices=self.intrinsic_matrices,
            image_paths=self.image_paths
        )
    
    def get_translations(self) -> np.ndarray:
        """Extract translation vectors from extrinsic matrices."""
        if self.extrinsic_matrices.shape[-1] == 4 and self.extrinsic_matrices.shape[-2] >= 3:
            return self.extrinsic_matrices[:, :3, 3]
        else:
            raise ValueError(f"Cannot extract translations from shape: {self.extrinsic_matrices.shape}")


class LocalizationResult(BaseModel):
    """Result from image localization (translation extraction only)."""
    
    translations: np.ndarray = Field(
        ..., 
        description="Translation vectors of shape (N, 3)"
    )
    image_paths: List[str] = Field(
        ..., 
        description="List of image paths corresponding to the translations"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def __len__(self) -> int:
        """Return the number of localizations."""
        return len(self.image_paths)
    
    def to_list(self) -> List[List[float]]:
        """Convert translations to list of [x, y, z] coordinates."""
        if self.translations.device.type == "cuda":
            translations = self.translations.cpu()
        else:
            translations = self.translations
            
        if hasattr(translations, 'numpy'):
            return translations.numpy().tolist()
        elif isinstance(translations, np.ndarray):
            return translations.tolist()
        else:
            return translations