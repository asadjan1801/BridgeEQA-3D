"""
Gemini-based PDF QA Generation Pipeline

This module uses the Google GenAI SDK directly (not OpenAI Agents SDK) to:
1. Map images to reference images in reports
2. Extract condition ratings from inspection reports
3. Generate QA pairs for bridge inspections

Uses structured outputs with Pydantic models and supports both sync and async operations.
"""

import json
import os
from pathlib import Path
import asyncio
import logfire
from google import genai
from google.genai.types import Part
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError

from .pdf_data_schemas import (
    ImageReferencePair,
    BridgeInspectionExtraction,
    QAPair,
    PDFGroundedQAGeneratedResponse,
)
from .qa_gen_prompt import QA_GENERATION_INSTRUCTIONS

# Global semaphore to limit concurrent Gemini API calls
_gemini_semaphore = asyncio.Semaphore(10)  # Only allow 2 concurrent API calls


def diagnose_gemini_response(response, context: str = ""):
    """
    Diagnose why response.parsed is None and log detailed error information.

    Args:
        response: Gemini API response object
        context: Context string to identify where this was called from

    Returns:
        dict with diagnostic information
    """
    diagnosis = {"context": context, "issue": "unknown"}

    # Check 1: Prompt-level blocking
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            diagnosis["issue"] = "prompt_blocked"
            diagnosis["block_reason"] = str(response.prompt_feedback.block_reason)
            if hasattr(response.prompt_feedback, 'safety_ratings'):
                diagnosis["safety_ratings"] = str(response.prompt_feedback.safety_ratings)
            logfire.error(f"{context} - Prompt blocked: {diagnosis['block_reason']}")
            return diagnosis

    # Check 2: No candidates returned
    if not hasattr(response, 'candidates') or not response.candidates:
        diagnosis["issue"] = "no_candidates"
        diagnosis["reason"] = "Prompt was blocked or invalid"
        logfire.error(f"{context} - No candidates returned")
        return diagnosis

    candidate = response.candidates[0]

    # Check 3: Safety blocking
    if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 'SAFETY':
        diagnosis["issue"] = "safety_blocked"
        diagnosis["finish_reason"] = str(candidate.finish_reason)
        if hasattr(candidate, 'safety_ratings'):
            diagnosis["safety_ratings"] = str(candidate.safety_ratings)
        logfire.error(f"{context} - Blocked by safety filter: {diagnosis.get('safety_ratings', 'unknown')}")
        return diagnosis

    # Check 4: Token limit exceeded
    if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 'MAX_TOKENS':
        diagnosis["issue"] = "token_limit_exceeded"
        diagnosis["finish_reason"] = str(candidate.finish_reason)
        logfire.error(f"{context} - Token limit exceeded (finish_reason: MAX_TOKENS)")
        return diagnosis

    # Check 5: Other finish reasons
    if hasattr(candidate, 'finish_reason') and candidate.finish_reason not in ['STOP', None]:
        diagnosis["issue"] = "unexpected_finish_reason"
        diagnosis["finish_reason"] = str(candidate.finish_reason)
        logfire.error(f"{context} - Unexpected finish reason: {diagnosis['finish_reason']}")
        return diagnosis

    # Check 6: Content exists but parsing failed (Pydantic/JSON error)
    if hasattr(candidate, 'content') and candidate.content:
        try:
            raw_text = response.text if hasattr(response, 'text') else None
            diagnosis["issue"] = "parsing_failed"
            diagnosis["reason"] = "Likely Pydantic validation error or invalid JSON (silently suppressed by SDK)"
            diagnosis["raw_text_preview"] = raw_text[:500] if raw_text else "No text available"
            logfire.error(f"{context} - Parsing failed. Raw text preview: {diagnosis['raw_text_preview']}")
        except Exception as e:
            diagnosis["text_access_error"] = str(e)
            logfire.error(f"{context} - Parsing failed and cannot access response.text: {e}")
        return diagnosis

    logfire.error(f"{context} - Unknown reason for None response.parsed")
    return diagnosis


@retry(
    retry=retry_if_exception_type((ClientError, Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=lambda retry_state: logfire.info(f"Rate limited, retrying in {retry_state.next_action.sleep} seconds...")
)
async def make_image_reference_mappings(
    report_gs_uri: str,
    image_gs_uris: list[str],
    output_path: str = None,
    model_name: str = "gemini-2.5-pro",
    use_cache: bool = True,
) -> list[ImageReferencePair]:
    """
    Create mappings between images and reference images in the report.

    Args:
        report_gs_uri: GCS URI to the PDF report (gs://bucket/path/to/report.pdf)
        image_gs_uris: List of GCS URIs to images
        output_path: Path to save results (optional)
        model_name: Gemini model to use
        use_cache: Whether to use cached results

    Returns:
        List of ImageReferencePair objects
    """
    # Check for cached results
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "image_reference_mappings.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                image_reference_mappings = [ImageReferencePair.model_validate(item) for item in cached_data]
                # only return if the image reference mappings are not empty
                if len(image_reference_mappings) > 0:
                    return image_reference_mappings

    # Initialize Gemini client
    client = genai.Client(
        vertexai=True,
        project=os.getenv("VERTEXAI_PROJECT"),
        location="global"
    )

    # Build prompt with images and report
    prompt_parts = [
        "You are a helpful assistant that makes mappings between provided images and reference images mentioned in the inspection report.\n\n",
        "For each image, identify which reference images (figures/photos) in the report it corresponds to, including the page number.\n\n",
        "Images:\n"
    ]

    # Add each image
    for i, image_uri in enumerate(image_gs_uris):
        image_name = Path(image_uri).name
        prompt_parts.extend([
            f"Image {i+1}: {image_name}\n",
            Part.from_uri(file_uri=image_uri, mime_type="image/jpeg"),
            "\n"
        ])

    # Add report
    prompt_parts.extend([
        "\nInspection Report:\n",
        Part.from_uri(file_uri=report_gs_uri, mime_type="application/pdf")
    ])

    logfire.info(f"Running image reference mapping with {len(image_gs_uris)} images")

    # Call Gemini with structured output
    async_client = client.aio
    try:
        # Use semaphore to limit concurrent API calls
        async with _gemini_semaphore:
            response = await async_client.models.generate_content(
                model=model_name,
                contents=prompt_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[ImageReferencePair],
                },
            )

        # Parse structured output
        result: list[ImageReferencePair] = response.parsed

        # Handle None response
        if result is None:
            diagnose_gemini_response(response, "make_image_reference_mappings")
            result = []
    finally:
        # Close the async client to avoid unclosed session warnings
        await async_client.aclose()

    logfire.info(
        f"Mapping completed successfully, generated {len(result) if result else 0} pairs"
    )

    # Save results if output_path provided
    if output_path:
        output_file = Path(output_path).parent / "image_reference_mappings.json"
        with open(output_file, "w") as f:
            json.dump(
                [pair.model_dump() for pair in result] if result else [], f, indent=2
            )

    return result


@retry(
    retry=retry_if_exception_type((ClientError, Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=lambda retry_state: logfire.info(f"Rate limited, retrying in {retry_state.next_action.sleep} seconds...")
)
async def extract_condition_ratings(
    report_gs_uri: str,
    image_reference_mappings: list[ImageReferencePair] = None,
    output_path: str = None,
    model_name: str = "gemini-2.5-pro",
    use_cache: bool = True,
) -> BridgeInspectionExtraction:
    """
    Extract condition ratings from the inspection report.

    Args:
        report_gs_uri: GCS URI to the PDF report
        image_reference_mappings: Pre-generated image reference mappings for consistency
        output_path: Path to save results (optional)
        model_name: Gemini model to use
        use_cache: Whether to use cached results

    Returns:
        BridgeInspectionExtraction object
    """
    # Check for cached results
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "condition_ratings.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            return BridgeInspectionExtraction.model_validate(cached_data)

    # Initialize Gemini client
    client = genai.Client(
        vertexai=True,
        project=os.getenv("VERTEXAI_PROJECT"),
        location="global"
    )

    # Build prompt
    prompt = [
        "You are a helpful assistant that extracts condition ratings from bridge inspection reports.\n\n",
        "Extract all relevant information including:\n",
        "- Agency name\n",
        "- Bridge Identification Number (BIN)\n",
        "- Inspection type and date\n",
        "- Component condition ratings (NBI scores)\n",
        "- Element-level condition states\n\n",
    ]

    # Add image reference mappings if available
    if image_reference_mappings:
        mappings_text = json.dumps([m.model_dump() for m in image_reference_mappings], indent=2)
        prompt.extend([
            "## Image Reference Mappings:\n",
            "Use these mappings to populate the reference_images field in condition observations.\n",
            "These show which images correspond to reference names in the document:\n",
            mappings_text,
            "\n\n",
        ])

    prompt.extend([
        "Inspection Report:\n",
        Part.from_uri(file_uri=report_gs_uri, mime_type="application/pdf")
    ])

    logfire.info("Running extraction")

    # Call Gemini with structured output
    async_client = client.aio
    try:
        # Use semaphore to limit concurrent API calls
        async with _gemini_semaphore:
            response = await async_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": BridgeInspectionExtraction,
                },
            )

        # Parse structured output
        result: BridgeInspectionExtraction = response.parsed

        # Handle None response
        if result is None:
            diagnose_gemini_response(response, "extract_condition_ratings")
            result = BridgeInspectionExtraction(
                agency="Unknown",
                bin="Unknown",
                inspection_type="Unknown",
                inspection_date="Unknown",
                component_ratings=[],
                element_ratings=[]
            )
    finally:
        # Close the async client to avoid unclosed session warnings
        await async_client.aclose()

    logfire.info("Extraction completed successfully")

    # Save results if output_path provided
    if output_path:
        output_file = Path(output_path).parent / "condition_ratings.json"
        with open(output_file, "w") as f:
            json.dump(result.model_dump() if result else {}, f, indent=2)

    return result


@retry(
    retry=retry_if_exception_type((ClientError, Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=lambda retry_state: logfire.info(f"Rate limited, retrying in {retry_state.next_action.sleep} seconds...")
)
async def generate_qa_pairs(
    report_gs_uri: str,
    image_gs_uris: list[str],
    image_reference_mappings: list[ImageReferencePair],
    condition_ratings: BridgeInspectionExtraction,
    output_path: str = None,
    model_name: str = "gemini-2.5-pro",
    use_cache: bool = True,
) -> list[QAPair]:
    """
    Generate QA pairs for bridge inspection based on images and report.

    Args:
        report_gs_uri: GCS URI to the PDF report
        image_gs_uris: List of GCS URIs to images
        image_reference_mappings: Pre-generated image reference mappings
        condition_ratings: Extracted condition ratings with verbatim observations
        output_path: Path to save results (optional)
        model_name: Gemini model to use
        use_cache: Whether to use cached results

    Returns:
        List of QAPair objects
    """
    # Check for cached results
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "qa_pairs.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            # Handle both formats: list[QAPair] or PDFGroundedQAGeneratedResponse
            if isinstance(cached_data, dict) and "qa_pairs" in cached_data:
                # Full response format
                qa_pairs_data = cached_data["qa_pairs"]
            elif isinstance(cached_data, list):
                # List format
                qa_pairs_data = cached_data
            else:
                logfire.warn(f"Unexpected cache format, regenerating QA pairs")
                qa_pairs_data = None

            if qa_pairs_data is not None:
                return [QAPair.model_validate(item) for item in qa_pairs_data]

    # Initialize Gemini client
    client = genai.Client(
        vertexai=True,
        project=os.getenv("VERTEXAI_PROJECT"),
        location="global"
    )

    # Build comprehensive prompt
    instructions = QA_GENERATION_INSTRUCTIONS

    # Add image reference mappings
    mappings_text = json.dumps([m.model_dump() for m in image_reference_mappings], indent=2)

    # Add condition ratings as primary source for verbatim answers
    condition_ratings_text = json.dumps(condition_ratings.model_dump(), indent=2)

    prompt_parts = [
        instructions,
        "\n\n## Extracted Condition Ratings (PRIMARY SOURCE FOR VERBATIM ANSWERS):\n",
        "Use the observations from this structured data as your primary source for answer text.\n",
        "Each ComponentRating and ElementRating contains:\n",
        "- location: Specific bridge location (e.g., 'Abutment 1', 'Pier 2')\n",
        "- observation: VERBATIM text from the inspection report\n",
        "- reference_images: Images that show this condition\n\n",
        condition_ratings_text,
        "\n\n## Image Reference Mappings:\n",
        "The following mappings show ALL available images extracted from the PDF and their corresponding reference names in the document.\n",
        "Use these mappings to understand which images are available and reference them by their document names (e.g., 'Figure 3.2', 'Photo 5').\n",
        mappings_text,
        "\n\nEnsure references to images align with reference image names in the mappings above.\n\n",
        "Inspection Report:\n",
        Part.from_uri(file_uri=report_gs_uri, mime_type="application/pdf"),
        "\n\nNote: The actual images are not provided directly. Use the Image Reference Mappings above to understand "
        "which images exist and reference them by their document names as shown in the mappings.\n"
    ]

    logfire.info(f"Running QA generation with {len(image_reference_mappings)} image mappings")

    # Call Gemini with structured output
    async_client = client.aio
    try:
        # Use semaphore to limit concurrent API calls
        async with _gemini_semaphore:
            try:
                logfire.debug(f"prompt_parts: {prompt_parts}")
                response = await async_client.models.generate_content(
                    model=model_name,
                    contents=prompt_parts,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[QAPair],
                    },
                )
                # Parse structured output
                result: list[QAPair] = response.parsed
            except Exception as e:
                logfire.error(f"Error generating QA pairs: {e}")
                logfire.error("Trying again with gemini-2.5-pro")
                response = await async_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt_parts,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": list[QAPair],
                    },
                )
                result: list[QAPair] = response.parsed

        # Handle None response
        if result is None:
            diagnose_gemini_response(response, "generate_qa_pairs")
            result = []
    finally:
        # Close the async client to avoid unclosed session warnings
        await async_client.aclose()

    logfire.info(
        f"QA generation completed, created {len(result) if result else 0} Q/A pairs"
    )

    # Save results if output_path provided
    if output_path:
        output_file = Path(output_path).parent / "qa_pairs.json"
        with open(output_file, "w") as f:
            json.dump(
                [pair.model_dump() for pair in result] if result else [], f, indent=2
            )

    return result


async def pdf_qa_gen(
    report_gs_uri: str,
    image_gs_uris: list[str],
    output_path: str,
    model_name: str = "gemini-2.5-pro",
    use_cache: bool = True,
    only_generate_qa_pairs: bool = False,
) -> PDFGroundedQAGeneratedResponse:
    """
    Main pipeline to generate QA pairs from PDF report and images.

    Args:
        report_gs_uri: GCS URI to the PDF report (gs://bucket/path/to/report.pdf)
        image_gs_uris: List of GCS URIs to images
        output_path: Path to save combined results
        model_name: Gemini model to use
        use_cache: Whether to use cached results
        only_generate_qa_pairs: Whether to only generate QA pairs and skip all other steps

    Returns:
        PDFGroundedQAGeneratedResponse object
    """
    logfire.info("Starting PDF QA generation pipeline")

    async def generate_pdf_qa_gen():
        # First generate image reference mappings
        image_reference_mappings = await make_image_reference_mappings(
            report_gs_uri, image_gs_uris, output_path, model_name, use_cache
        )

        # Then extract condition ratings with the mappings for consistency
        condition_ratings = await extract_condition_ratings(
            report_gs_uri, image_reference_mappings, output_path, model_name, use_cache
        )

        assert len(image_reference_mappings) > 0, "Image reference mappings are empty"
        assert len(condition_ratings.condition_ratings) > 0, "Condition ratings are empty"

        # Generate QA pairs (needs mappings and condition ratings from previous step)
        if only_generate_qa_pairs:
            qa_pairs = await generate_qa_pairs(
                report_gs_uri,
                image_gs_uris,
                image_reference_mappings,
                condition_ratings,
                output_path,
                model_name,
                False,
            )
        else:
            qa_pairs = await generate_qa_pairs(
                report_gs_uri,
                image_gs_uris,
                image_reference_mappings,
                condition_ratings,
                output_path,
                model_name,
                use_cache,
            )

        return PDFGroundedQAGeneratedResponse(
            image_reference_mappings=image_reference_mappings,
            condition_ratings=condition_ratings,
            qa_pairs=qa_pairs,
        )

    pdf_qa_gen_result = await generate_pdf_qa_gen()

    with open(output_path, "w") as f:
        json.dump(pdf_qa_gen_result.model_dump(), f, indent=2)

    logfire.info("PDF QA generation pipeline completed successfully")

    logfire.warning(
        "QA pairs should be reviewed to ensure answers align with the inspection report in terms of referenced images and reasoning."
    )

    return pdf_qa_gen_result