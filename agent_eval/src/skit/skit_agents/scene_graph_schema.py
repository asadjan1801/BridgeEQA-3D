"""Enhanced scene graph models for multi-agent scene understanding with spatial context."""

from pydantic import BaseModel, Field
from typing import List, Optional


class SpatialCoordinates(BaseModel):
    """Spatial coordinates from VGGt or other vision-geometry models."""

    x: float = Field(description="X coordinate in 3D space")
    y: float = Field(description="Y coordinate in 3D space")
    z: float = Field(description="Z coordinate in 3D space")


class EdgeConnection(BaseModel):
    """Enhanced edge connection between nodes with descriptive relationships."""

    connected_to: str = Field(description="Name of the connected node")
    description_of_connection: str = Field(
        description="Detailed description of how nodes are connected"
    )


class Node(BaseModel):
    """Enhanced node in the scene graph with spatial context and detailed descriptions."""

    image_name: str = Field(description="Name of the image file")
    central_focus: str = Field(description="The main subject or focus of the image")
    image_description: str = Field(
        description="Detailed description of what's visible in the image"
    )
    edges: List[EdgeConnection] = Field(description="edge connections to other nodes")


class NodeWithCameraLocalization(BaseModel):
    """Enhanced node in the scene graph with spatial context and detailed descriptions."""

    image_name: str = Field(description="Name of the image file")
    central_focus: str = Field(description="The main subject or focus of the image")
    image_description: str = Field(
        description="Detailed description of what's visible in the image"
    )
    edges: List[EdgeConnection] = Field(description="edge connections to other nodes")
    spatial_coordinates: Optional[SpatialCoordinates] = Field(
        description="Spatial coordinates of the node calculated by camera pose estimation"
    )


class SceneGraph(BaseModel):
    """Complete enhanced scene graph with spatial context and multi-agent support."""

    nodes: List[Node] | List[NodeWithCameraLocalization] = Field(
        description="All scene nodes in the graph"
    )
