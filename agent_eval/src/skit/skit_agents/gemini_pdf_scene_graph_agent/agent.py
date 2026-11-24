"""
Gemini-based PDF Scene Graph Agent

This module uses the Google GenAI SDK directly (not OpenAI Agents SDK) to generate
scene graphs from bridge inspection PDFs and images stored in Google Cloud Storage.

Uses structured outputs with Pydantic models and supports async operations.
"""

from __future__ import annotations
from google import genai
from google.genai.types import Part
from skit.skit_agents.scene_graph_agent.scene_graph_schema import SceneGraph
from skit.pipelines.models import ExtractedImage
from pathlib import Path
from typing import List
import os
import logfire
from skit.pipelines.pdf_data_schemas import PDFGroundedQAGeneratedResponse
PDF_TO_SCENE_GRAPH_INSTRUCTIONS = """
**Agent Persona:** You are a highly accurate Bridge Inspection Structural Mapping Agent specializing in generating descriptive scene graphs (SGs) from visual inspection data and corresponding field notes.

**Task:** Your sole function is to transform bridge inspection photographs and associated descriptive text into a structured Scene Graph that meticulously documents component identification, observable physical attributes, and spatial/functional relationships. The resulting Scene Graph will be used by a separate analysis agent strictly as a map to establish locality and structural context.

**Critical Constraint:** The output **MUST NOT** include any numerical condition ratings, Condition State codes (CS 1, CS 2, CS 3, CS 4), or NBI codes (0-9, G/F/P codes) associated with component or element condition. All condition information must be conveyed using qualitative descriptive terminology only.

### Scene Graph Schema Requirements

Generate output conforming to the following structure for each identified feature or element:

#### I. Node Identification (Component & Viewpoint)

*   **Format:** Identify the specific bridge element, its material, and the physical location/viewpoint captured in the image/data.
*   **Examples based on sources:** "Deck Slab (RC) at Span 1 midspan", "Steel Truss Bottom Chord, upstream side near Abutment 1 bearing", "Reinforced Concrete Abutment 2 wingwall, western face".

#### II. Image/Feature Description (Node Attributes)

Provide a detailed description structured with relevant information such as Component Identification, Material, Construction Detail, Observable Conditions, Physical, Environmental Context.

**Include these critical descriptive elements drawn from bridge inspection terminology:**

*   **Component Identification:** Specify type, material (e.g., Prestressed Concrete Box Girder, Steel Open Girder/Beam, RC Slab, Masonry Abutment), and specific location (e.g., bearing seat, flange/web, floor system, fascia).
*   **Observable Conditions/Defects (Qualitative Descriptions ONLY):**
    *   **Concrete Defects:** Describe cracking (e.g., "map cracking", "fine cracks", "vertical cracking"), presence of efflorescence or rust staining, spalling or delamination, and exposed or corroding reinforcement (rebar).
    *   **Steel/Paint Defects:** Describe corrosion (e.g., "freckled rust", "moderate rust scaling"), paint condition (e.g., "scaling paint", "peeling/bubbling"), and section loss (specify location, if measurable or apparent).
    *   **Fastener/Connection Issues:** Note loose, missing, or broken bolts/rivets/connectors.
    *   **Movement/Alignment:** Describe observable sag/deflection, misalignment (vertical/horizontal), or restricted movement (e.g., debris restricting bearing movement).
*   **Water/Channel/Scour:** Describe scour/erosion present at foundations, leakage from joints, debris accumulation (e.g., gravel/dirt/beaver dams), and resulting issues like foundation undermining or soil washout.
*   **Extent Descriptors (Use these terms to quantify scope):** isolated / scattered / widespread / throughout [As per user example].
*   **Context and Reference Elements:**
    *   Previous repairs (patches/overlays and visual performance).
    *   Safety features documented (e.g., condition of metal bridge railing, traffic control devices, vertical clearance signs).
    *   Scale references (e.g., visible objects providing size context).

#### III. Edge Connections (Relational Semantics)

Define the connections between the identified nodes based on the relationships observed in the image set:

**A. Spatial Relationships (Locality Mapping):**
*   **"adjacent\_to"** - Neighboring views or physical components (e.g., Deck Fascia adjacent\_to Bridge Railing).
*   **"overview\_of"** - A broader context view encompassing smaller elements.
*   **"opposite\_of"** - Opposing view/side (e.g., Upstream side opposite\_of Downstream side).
*   **"continuation\_of"** - Extends the view along a length (e.g., Span 1 continuation\_of Abutment 1).

**B. Hierarchical Relationships (Structural Dependency):**
*   **"contains"** - Encompasses smaller elements (e.g., Deck contains Wearing Surfaces).
*   **"supports"** - Defines a load-bearing relationship (e.g., Abutment supports Girder/Beam).

**C. Functional/Causal Relationships (Observable Impact):**
*   **"affects"** - Causal relationship where one condition or component impacts another (e.g., Joint leakage affects bearing condition, Corrosion affects load capacity).
*   **"similar\_to"** - Comparable conditions or features observed across different elements (e.g., Span 1 abutment cracking similar\_to Span 2 abutment cracking).
"""


class PDFSceneGraphAgent:
    """
    Generates scene graphs from bridge inspection PDFs and images using Gemini.

    This version uses:
    - Google GenAI SDK (not OpenAI Agents SDK)
    - GCS URIs for files (expects files already uploaded to GCS)
    - Structured outputs with Pydantic models
    """

    def __init__(
        self,
        extracted_images: List[ExtractedImage],
        model_name: str = "gemini-2.5-pro",
        project: str = None,
        location: str = None,
    ):
        """
        Initialize the PDF Scene Graph Agent.

        Args:
            extracted_images: List of ExtractedImage objects with GCS URIs
            model_name: Gemini model to use (default: gemini-2.5-pro)
            project: GCP project ID (defaults to env VERTEXAI_PROJECT)
            location: GCP location (defaults to env VERTEXAI_LOCATION)
        """
        self.model_name = model_name
        self.extracted_images = extracted_images

        # Get project and location from environment if not provided
        self.project = project or os.getenv("VERTEXAI_PROJECT")
        self.location = location or os.getenv("VERTEXAI_LOCATION", "us-south1")

        # Initialize Gemini client
        self.client = genai.Client(
            vertexai=True, project=self.project, location=self.location
        )

    async def generate_scene_graph(self, pdf_gs_uri: str,extracted_images: List[ExtractedImage], full_qa_response: PDFGroundedQAGeneratedResponse) -> SceneGraph:
        """
        Generate a scene graph from a PDF and its extracted images.

        Args:
            pdf_gs_uri: GCS URI to the PDF (gs://bucket/path/to/file.pdf)

        Returns:
            SceneGraph object with nodes and edges

        Note:
            All images in extracted_images should have cloud_download_url
            populated with GCS URIs (gs://bucket/path/to/image.jpg)
        """
        self.extracted_images = extracted_images
        # Build prompt parts
        prompt_parts = [
            "\n\nInspection Report:\n",
            Part.from_uri(file_uri=pdf_gs_uri, mime_type="application/pdf"),
            "\n\nImages:\n",
        ]

        # Add each image with its name annotation
        for image in self.extracted_images:
            assert (
                image.cloud_download_url is not None
            ), f"Image {image.image_path} has no GCS URI"

            # Determine MIME type from file extension
            image_path = Path(image.image_path)
            mime_type = self._get_mime_type(image_path.suffix.lower())

            # Add image with name annotation
            prompt_parts.extend(
                [
                    f"\nImage name for scene graph node: {image.image_path}\n",
                    Part.from_uri(
                        file_uri=image.cloud_download_url, mime_type=mime_type
                    ),
                ]
            )

        # Add image reference mappings (OUTSIDE the loop)
        prompt_parts.append("\n\n## Image Reference Mappings:\n")
        for image_reference_mapping in full_qa_response.image_reference_mappings:
            prompt_parts.append(f"Image {image_reference_mapping.image_name} corresponds to reference image {image_reference_mapping.reference_images} on page {image_reference_mapping.reference_image_page}\n")

        # Add instructions (OUTSIDE the loop)
        prompt_parts.append("\n" + PDF_TO_SCENE_GRAPH_INSTRUCTIONS + f"\nOnly {len(self.extracted_images)} images should be used to generate the scene graph.")

        # Call Gemini with structured output
        async_client = self.client.aio
        try:
            response = await async_client.models.generate_content(
                model=self.model_name,
                contents=prompt_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": SceneGraph,
                },
            )

            scene_graph: SceneGraph = response.parsed

            # Check if number of images and nodes match
            if len(self.extracted_images) != len(scene_graph.nodes):
                logfire.error(f"Node count mismatch with {self.model_name}: received {len(scene_graph.nodes)} nodes but have {len(self.extracted_images)} images. Retrying with gemini-2.5-pro...")

                # Retry with gemini-2.5-pro
                response = await async_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt_parts,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": SceneGraph,
                    },
                )
                scene_graph: SceneGraph = response.parsed

                # Final check
                if len(self.extracted_images) != len(scene_graph.nodes):
                    logfire.warning(f"Node count still mismatched after retry: {len(scene_graph.nodes)} nodes vs {len(self.extracted_images)} images. Proceeding anyway.")

        finally:
            await async_client.aclose()

        return scene_graph

    def _get_mime_type(self, extension: str) -> str:
        """
        Get MIME type from file extension.

        Args:
            extension: File extension (e.g., '.jpg', '.png')

        Returns:
            MIME type string
        """
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(extension, "image/jpeg")  # Default to JPEG