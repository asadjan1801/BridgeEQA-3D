"""
PDF processing pipeline that extracts images from PDFs and generates scene graphs.
Similar to image_folder_pipeline but handles PDF input instead of image folders.
"""

from __future__ import annotations
import pymupdf  # PyMuPDF
import shutil
from pathlib import Path
import json
from skit.skit_agents.scene_graph_schema import SceneGraph
# from skit.skit_agents.pdf_scene_graph_agent.agent import PDFSceneGraphAgent
from skit.skit_agents.gemini_pdf_scene_graph_agent.agent import PDFSceneGraphAgent
from skit.files.google_cloud_storage import GCSFileManager
from skit.pipelines.models import ExtractedImage
from skit.pipelines.gemini_pdf_qa_gen import pdf_qa_gen
import hashlib
import os
import logfire

async def extract_images_from_pdf(
    pdf_path: str,
    output_folder: str,
    gcs_manager: GCSFileManager,
    min_width: int = 200,
    min_height: int = 200,
) -> list[ExtractedImage]:
    """
    Extract images from a PDF file and save them to the output folder and upload them to openai.

    Args:
        pdf_path: Path to the PDF file
        output_folder: Directory to save extracted images
        min_width: Minimum width for extracted images (filters out tiny images)
        min_height: Minimum height for extracted images

    Returns:
        List of paths to extracted images
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    extracted_images = []

    try:
        # Open the PDF
        doc = pymupdf.open(pdf_path)

        image_counter = 0
        seen_images = set()

        for page_index in range(len(doc)):
            # skip the first 2 pages
            if page_index < 2:
                continue
            page = doc[page_index]
            # Get all images on the page
            image_list = page.get_images()

            for image_index, img in enumerate(image_list):
                # Get the XREF of the image
                xref = img[0]

                # Extract the image
                pix = pymupdf.Pixmap(doc, xref)

                # Check image dimensions
                if pix.width < min_width or pix.height < min_height:
                    pix = None
                    continue

                # Convert CMYK to RGB if necessary
                if pix.n - pix.alpha > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                # Save to temp file first to calculate hash
                temp_filename = f"temp_{page_index}_{image_index}.png"
                temp_path = output_path / temp_filename
                pix.save(str(temp_path))

                # Calculate hash (matching GCS convention - first 32 chars)
                with open(temp_path, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()[:32]

                # Check if this image already exists (deduplication)
                final_filename = f"{content_hash}.png"
                final_path = output_path / final_filename

                if content_hash in seen_images:
                    logfire.info(f"Image {content_hash} already exists, skipping")
                    temp_path.unlink()  # Delete temp file
                    continue

                # Rename temp file to hash-based name
                temp_path.rename(final_path)
                seen_images.add(content_hash)
                image_path = final_path

                # Upload image to GCS
                image_gs_uri = gcs_manager.upload_file(image_path)

                extracted_images.append(
                    ExtractedImage(
                        image_path=str(image_path),
                        cloud_download_url=image_gs_uri
                    )
                )
                image_counter += 1

                pix = None

        doc.close()
        logfire.info(f"Extracted {image_counter} images from {pdf_path}")

    except Exception as e:
        logfire.error(f"Error extracting images from PDF: {e}")
        raise

    return extracted_images


async def pdf_pipeline(
    pdf_path: str,
    scene_id: str,
    output_folder: str = "processed_scenes",
    regenerate_scene_graph: bool = False,
    model_name: str = "gemini-2.5-pro",
    min_image_width: int = 100,
    min_image_height: int = 100,
    use_cache: bool = True,
    only_generate_qa_pairs: bool = False,
) -> SceneGraph:
    """
    Pipeline to process a PDF file: extract images, generate scene graph, and optionally copy QA pairs.

    Args:
        pdf_path: Path to the PDF file
        scene_id: Unique identifier for the scene, for example the name of the pdf file
        output_folder: Output directory for processed data
        qa_file: Optional path to JSONL file with QA pairs
        regenerate_scene_graph: Force regeneration of scene graph even if it exists
        model_name: Model to use for scene graph generation
        min_image_width: Minimum width for extracted images
        min_image_height: Minimum height for extracted images
        use_cache: Whether to use cache for image extraction
        only_generate_qa_pairs: Whether to only generate QA pairs and skip all other steps
    Returns:
        Generated SceneGraph object
    """

    scene_id = str(scene_id) if isinstance(scene_id, int) else scene_id
    scene_path = Path(output_folder) / scene_id
    images_path = scene_path / "images"
    scene_graph_path = scene_path / "scene_graph.json"
    
    # if cache is disabled, clear the output folder
    # if use_cache is False:
    #     logfire.info(f"Cache disabled, clearing cache in {scene_path}")
    #     # clear cache
    #     if scene_path.exists():
    #         shutil.rmtree(scene_path)

    # make output folder if it doesn't exist
    scene_path.mkdir(parents=True, exist_ok=True)
    # Initialize GCS manager
    bucket_name = os.getenv("GOOGLE_BUCKET_NAME")
    gcs_manager = GCSFileManager(bucket_name=bucket_name)

    # save a copy of the pdf to the output folder
    pdf_file_name = Path(pdf_path).name
    pdf_save_path = scene_path / pdf_file_name
    if not pdf_save_path.exists():
        shutil.copy(pdf_path, pdf_save_path)
    pdf_path = pdf_save_path

    # Extract images from PDF
    extracted_images = await extract_images_from_pdf(
        pdf_path=pdf_save_path,
        output_folder=str(images_path),
        gcs_manager=gcs_manager,
        min_width=min_image_width,
        min_height=min_image_height,
    )

    # Step 1: Extract images from PDF if needed
    if not extracted_images:
        logfire.warning(f"images_path exists: {images_path.exists()}")
        logfire.warning(f"regenerate_scene_graph set to {regenerate_scene_graph}")
        logfire.warning(f"No images extracted from PDF: {pdf_save_path}")
        return None

    assert len(extracted_images) > 0, "No images extracted from PDF"
    for image in extracted_images:
        assert image.cloud_download_url is not None, "Image cloud download url is None"
    qa_path = str(scene_path / "full_response_qa_pairs.json")

    full_qa_response = None

    # Upload PDF and collect image GCS URIs (needed for both QA generation and scene graph)
    report_gs_uri = gcs_manager.upload_file(pdf_save_path)
    image_gs_uris = [img.cloud_download_url for img in extracted_images]

    # Generate QA pairs
    if only_generate_qa_pairs:
        # Force regeneration of QA pairs
        logfire.warn(f"Only generating QA pairs (forced regeneration) for {pdf_path} and {images_path} to {qa_path}")
        full_qa_response = await pdf_qa_gen(
            report_gs_uri=report_gs_uri,
            image_gs_uris=image_gs_uris,
            output_path=qa_path,
            model_name=model_name,
            use_cache=False,  # Force regeneration when only_generate_qa_pairs is True
            only_generate_qa_pairs=True,
        )
    elif not Path(qa_path).exists() or not use_cache:
        # Generate QA pairs if they don't exist or cache is disabled
        logfire.info(f"Generating QA pairs for {pdf_path} and {images_path} to {qa_path}")
        full_qa_response = await pdf_qa_gen(
            report_gs_uri=report_gs_uri,
            image_gs_uris=image_gs_uris,
            output_path=qa_path,
            model_name=model_name,
            use_cache=use_cache,
        )
    else:
        # Load existing QA response for scene graph generation
        if Path(qa_path).exists():
            from skit import PDFGroundedQAGeneratedResponse
            with open(qa_path, "r") as f:
                qa_data = json.load(f)
            full_qa_response = PDFGroundedQAGeneratedResponse.model_validate(qa_data)
    # checks if the qa pairs or image reference mappings are empty, if so, regenerate them
    if len(full_qa_response.qa_pairs) == 0 or len(full_qa_response.image_reference_mappings) == 0:
        logfire.warn("QA pairs or image reference mappings are empty for {pdf_path}, attempting to regenerate QA pairs", pdf_path=pdf_path)
        full_qa_response = await pdf_qa_gen(
            report_gs_uri=report_gs_uri,
            image_gs_uris=image_gs_uris,
            output_path=qa_path,
            model_name=model_name,
            use_cache=False,
        )
        
    # Step 2: Generate or load scene graph
    # agent = PDFSceneGraphAgent(model_name=model_name, extracted_images=extracted_images)
    agent = PDFSceneGraphAgent(extracted_images=extracted_images)

    scene_graph = None
    if scene_graph_path.exists() and not regenerate_scene_graph:
        with open(scene_graph_path, "r") as f:
            scene_dict = json.load(f)
        scene_graph = SceneGraph.model_validate(scene_dict)
        logfire.info(f"Loaded existing scene graph from {scene_graph_path}")
    else:
        try:
            logfire.info("Generating scene graph from PDF")
            # Upload PDF to GCS and get gs:// URI
            pdf_gs_uri = gcs_manager.upload_file(pdf_save_path)

            scene_graph = await agent.generate_scene_graph(pdf_gs_uri=pdf_gs_uri,extracted_images=extracted_images, full_qa_response=full_qa_response)
            # Save scene graph
            with open(scene_graph_path, "w") as f:
                json.dump(scene_graph.model_dump(), f, indent=2)
        except Exception as e:
            import traceback

            logfire.error(
                f"Error generating scene graph: {e}\n{traceback.format_exc()}",
                _exc_info=True,
            )
            return None

    

    return scene_graph


async def pdf_image_extraction_pipeline(
    pdf_path: str,
    scene_id: str,
    output_folder: str = "processed_scenes",
    regenerate_scene_graph: bool = False, # for backward compatibility
    model_name: str = "gemini-2.5-pro", # for backward compatibility
    min_image_width: int = 100,
    min_image_height: int = 100,
    use_cache: bool = True,
    only_generate_qa_pairs: bool = False, # for backward compatibility
) -> SceneGraph:
    """
    Pipeline to process a PDF file: extract images, generate scene graph, and optionally copy QA pairs.

    Args:
        pdf_path: Path to the PDF file
        scene_id: Unique identifier for the scene, for example the name of the pdf file
        output_folder: Output directory for processed data
        qa_file: Optional path to JSONL file with QA pairs
        regenerate_scene_graph: Force regeneration of scene graph even if it exists # for backward compatibility
        model_name: Model to use for scene graph generation
        min_image_width: Minimum width for extracted images
        min_image_height: Minimum height for extracted images
        use_cache: Whether to use cache for image extraction
        only_generate_qa_pairs: Whether to only generate QA pairs and skip all other steps # for backward compatibility
    Returns:
        Generated SceneGraph object
    """

    scene_id = str(scene_id) if isinstance(scene_id, int) else scene_id
    scene_path = Path(output_folder) / scene_id
    images_path = scene_path / "images"
    scene_graph_path = scene_path / "scene_graph.json"
    
    # if cache is disabled, clear the output folder
    # if use_cache is False:
    #     logfire.info(f"Cache disabled, clearing cache in {scene_path}")
    #     # clear cache
    #     if scene_path.exists():
    #         shutil.rmtree(scene_path)

    # make output folder if it doesn't exist
    scene_path.mkdir(parents=True, exist_ok=True)
    # Initialize GCS manager
    bucket_name = os.getenv("GOOGLE_BUCKET_NAME")
    gcs_manager = GCSFileManager(bucket_name=bucket_name)

    # save a copy of the pdf to the output folder
    pdf_file_name = Path(pdf_path).name
    pdf_save_path = scene_path / pdf_file_name
    shutil.copy(pdf_path, pdf_save_path)
    pdf_path = pdf_save_path

    # Extract images from PDF
    extracted_images = await extract_images_from_pdf(
        pdf_path=pdf_save_path,
        output_folder=str(images_path),
        gcs_manager=gcs_manager,
        min_width=min_image_width,
        min_height=min_image_height,
    )

    return extracted_images


async def local_pdf_pipeline(
    pdf_path: str,
    scene_id: str,
    output_folder: str,
    min_image_width: int = 200,
    min_image_height: int = 200,
    use_cache: bool = True,
) -> SceneGraph | None:
    """
    Process a PDF using a local vLLM-served model. No GCS needed.

    Requires: local_pdf_qa_gen.configure() called before use.
    """
    from skit.pipelines.local_pdf_qa_gen import pdf_qa_gen as local_pdf_qa_gen

    scene_path = Path(output_folder) / scene_id
    images_path = scene_path / "images"
    scene_graph_path = scene_path / "scene_graph.json"
    scene_path.mkdir(parents=True, exist_ok=True)

    # Copy PDF to output folder
    pdf_file_name = Path(pdf_path).name
    pdf_save_path = scene_path / pdf_file_name
    if not pdf_save_path.exists():
        shutil.copy(pdf_path, pdf_save_path)

    # Extract images locally (no GCS)
    extracted_images = await extract_images_from_pdf_local(
        pdf_path=str(pdf_save_path),
        output_folder=str(images_path),
        min_width=min_image_width,
        min_height=min_image_height,
    )

    if not extracted_images:
        logfire.warning(f"No images extracted from {pdf_save_path}")
        return None

    image_paths = [img.image_path for img in extracted_images]
    qa_path = str(scene_path / "full_response_qa_pairs.json")

    # Generate QA pairs using local model
    full_qa_response = await local_pdf_qa_gen(
        pdf_path=str(pdf_save_path),
        image_paths=image_paths,
        output_path=qa_path,
        use_cache=use_cache,
    )

    # Generate scene graph from extracted data
    from skit.skit_agents.scene_graph_schema import Node, EdgeConnection
    if scene_graph_path.exists() and use_cache:
        with open(scene_graph_path) as f:
            scene_graph = SceneGraph.model_validate(json.load(f))
        logfire.info(f"Loaded existing scene graph from {scene_graph_path}")
    else:
        # Build a deterministic scene graph from images + QA data
        nodes = []
        for i, img in enumerate(extracted_images):
            img_name = Path(img.image_path).name
            description = "Extracted from inspection report"
            central_focus = f"Bridge inspection image {i+1}"
            if full_qa_response and full_qa_response.image_reference_mappings:
                for mapping in full_qa_response.image_reference_mappings:
                    if mapping.image_name == img_name:
                        central_focus = mapping.reference_images
                        description = f"Page {mapping.reference_image_page}: {mapping.reference_images}"
                        break

            nodes.append(Node(
                image_name=img_name,
                central_focus=central_focus,
                image_description=description,
                edges=[],
            ))

        # Add edges between consecutive nodes
        for i in range(len(nodes) - 1):
            nodes[i].edges.append(EdgeConnection(
                connected_to=nodes[i + 1].image_name,
                description_of_connection="Sequential inspection image",
            ))

        scene_graph = SceneGraph(nodes=nodes)
        with open(scene_graph_path, "w") as f:
            json.dump(scene_graph.model_dump(), f, indent=2)
        logfire.info(f"Built scene graph with {len(nodes)} nodes from extracted images")

    return scene_graph


async def extract_images_from_pdf_local(
    pdf_path: str,
    output_folder: str,
    min_width: int = 200,
    min_height: int = 200,
) -> list[ExtractedImage]:
    """Extract images from PDF without GCS upload."""
    pdf_path_obj = Path(pdf_path)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

    extracted_images = []
    doc = pymupdf.open(pdf_path_obj)
    seen_images = set()

    for page_index in range(len(doc)):
        if page_index < 2:
            continue
        page = doc[page_index]
        for image_index, img in enumerate(page.get_images()):
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            if pix.width < min_width or pix.height < min_height:
                pix = None
                continue
            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            temp_path = output_path / f"temp_{page_index}_{image_index}.png"
            pix.save(str(temp_path))

            with open(temp_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()[:32]

            final_filename = f"{content_hash}.png"
            final_path = output_path / final_filename

            if content_hash in seen_images:
                temp_path.unlink()
                continue

            temp_path.rename(final_path)
            seen_images.add(content_hash)
            extracted_images.append(
                ExtractedImage(image_path=str(final_path), cloud_download_url="")
            )
            pix = None

    doc.close()
    logfire.info(f"Extracted {len(extracted_images)} images from {pdf_path_obj} (local)")
    return extracted_images