"""
Local-model PDF QA Generation Pipeline

Drop-in replacement for gemini_pdf_qa_gen.py that uses a local vLLM-served
model (e.g., Gemma 4 31B-it) via the OpenAI-compatible API.

Key differences from the Gemini version:
- PDF content provided as PyMuPDF-extracted text (no GCS URIs)
- Images sent as base64 data URIs (no Part.from_uri)
- JSON output parsed manually (no native structured output)
"""

import json
import re
import base64
import io
import asyncio
from pathlib import Path

import logfire
import pymupdf
from PIL import Image
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .pdf_data_schemas import (
    ImageReferencePair,
    BridgeInspectionExtraction,
    QAPair,
    PDFGroundedQAGeneratedResponse,
)
from skit.bridge_inspection_schemas import (
    NbiScore,
    ComponentRatings,
    ElementRating,
    ConditionObservation,
)
from .qa_gen_prompt import QA_GENERATION_INSTRUCTIONS

# Concurrency limiter — local GPU can only handle a few requests at once
_local_semaphore = asyncio.Semaphore(2)

# Reusable client — set once via configure()
_client: AsyncOpenAI | None = None
_model_name: str = "gemma4-base"

MAX_IMAGE_PIXELS = 1_003_520  # ~1 MP, keeps visual tokens manageable


def configure(base_url: str, model_name: str = "gemma4-base"):
    """Call once at startup to set the vLLM endpoint."""
    global _client, _model_name
    _client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    _model_name = model_name


def _extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"--- Page {i+1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def _image_to_base64_url(file_path: Path) -> str:
    """Resize and convert a local image to a base64 data URI."""
    img = Image.open(file_path).convert("RGB")
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"([\[\{][\s\S]*[\]\}])")


def _extract_json(text: str):
    """Pull the first JSON array or object out of model output."""
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(1))
    m = _BARE_JSON_RE.search(text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text)


async def _chat(messages: list[dict], max_tokens: int = 4096) -> str:
    """Send a chat completion to the local vLLM server and return text."""
    assert _client is not None, "Call local_pdf_qa_gen.configure() first"
    async with _local_semaphore:
        resp = await _client.chat.completions.create(
            model=_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    return resp.choices[0].message.content


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    before_sleep=lambda rs: logfire.info(f"Retrying image mapping in {rs.next_action.sleep}s..."),
)
async def make_image_reference_mappings(
    pdf_path: str,
    image_paths: list[str],
    output_path: str = None,
    use_cache: bool = True,
) -> list[ImageReferencePair]:
    """Map extracted images to reference figure names in the report."""
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "image_reference_mappings.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                if data:
                    return [ImageReferencePair.model_validate(item) for item in data]

    report_text = _extract_text_from_pdf(pdf_path)

    # Build message with images + report text
    content_parts = [
        {"type": "text", "text": (
            "You are a helpful assistant that maps extracted images to reference "
            "figures/photos in an inspection report.\n\n"
            "For each image, identify which reference image (figure/photo) in the "
            "report it corresponds to, and the page number.\n\n"
            "Images:\n"
        )},
    ]
    for i, img_path in enumerate(image_paths):
        img_name = Path(img_path).name
        content_parts.append({"type": "text", "text": f"Image {i+1}: {img_name}"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": _image_to_base64_url(Path(img_path))},
        })

    content_parts.append({"type": "text", "text": (
        f"\n\nInspection Report Text:\n{report_text}\n\n"
        "Return a JSON array where each element has:\n"
        '  {"image_name": "<filename>", "reference_images": "<Figure/Photo name>", "reference_image_page": <page_number>}\n'
        "Return ONLY the JSON array, no other text."
    )})

    logfire.info(f"Running local image reference mapping with {len(image_paths)} images")
    raw = await _chat([{"role": "user", "content": content_parts}], max_tokens=4096)

    try:
        parsed = _extract_json(raw)
        result = [ImageReferencePair.model_validate(item) for item in parsed]
    except Exception as e:
        logfire.error(f"Failed to parse image mappings: {e}\nRaw: {raw[:500]}")
        result = []

    if output_path:
        out_file = Path(output_path).parent / "image_reference_mappings.json"
        with open(out_file, "w") as f:
            json.dump([r.model_dump() for r in result], f, indent=2)

    return result


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    before_sleep=lambda rs: logfire.info(f"Retrying condition extraction in {rs.next_action.sleep}s..."),
)
async def extract_condition_ratings(
    pdf_path: str,
    image_reference_mappings: list[ImageReferencePair] = None,
    output_path: str = None,
    use_cache: bool = True,
) -> BridgeInspectionExtraction:
    """Extract condition ratings from the inspection report."""
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "condition_ratings.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return BridgeInspectionExtraction.model_validate(json.load(f))

    report_text = _extract_text_from_pdf(pdf_path)

    prompt = (
        "You are a helpful assistant that extracts condition ratings from bridge "
        "inspection reports.\n\n"
        "Extract all relevant information including:\n"
        "- Agency name\n"
        "- Bridge Identification Number (BIN)\n"
        "- Inspection type and date\n"
        "- Component condition ratings (NBI scores)\n"
        "- Element-level condition states\n\n"
    )

    if image_reference_mappings:
        mappings_text = json.dumps([m.model_dump() for m in image_reference_mappings], indent=2)
        prompt += (
            "## Image Reference Mappings:\n"
            "Use these mappings to populate the reference_images field in condition observations.\n"
            f"{mappings_text}\n\n"
        )

    prompt += (
        f"Inspection Report:\n{report_text}\n\n"
        "Return a JSON object with this structure:\n"
        '{"agency": "...", "bin": "...", "inspection_type": "Routine|In-Depth|Special|Other|Unknown", '
        '"inspection_date": "YYYY-MM-DD", '
        '"condition_ratings": [{"component": "...", "nbi_rating": N, "location": "...", '
        '"observation": "verbatim text", "reference_images": ["img.png"]}], '
        '"element_ratings": [{"element": "...", "condition_state": N, "quantity": N, '
        '"location": "...", "observation": "verbatim text", "reference_images": ["img.png"]}]}\n'
        "Return ONLY the JSON object, no other text."
    )

    logfire.info("Running local condition rating extraction")
    raw = await _chat([{"role": "user", "content": prompt}], max_tokens=8192)

    try:
        parsed = _extract_json(raw)
        # Build proper Pydantic objects from simplified model output
        condition_ratings_list = []
        for cr in parsed.get("condition_ratings", []):
            component = cr.get("component", "Unknown")
            # Map to valid ComponentName
            valid_names = ["Deck", "Superstructure", "Substructure", "Channel", "Culverts", "Approach"]
            comp_name = next((n for n in valid_names if n.lower() in component.lower()), "Deck")
            rating = cr.get("nbi_rating") or cr.get("rating")
            obs = cr.get("observation") or ""
            loc = cr.get("location") or "General"
            refs = cr.get("reference_images") or []
            condition_ratings_list.append(ComponentRatings(
                component_prefix="",
                component_suffix="",
                component_name=comp_name,
                component_rating=NbiScore(score=int(rating) if rating is not None else None),
                condition_observation=ConditionObservation(
                    location=loc, observation=obs, reference_images=refs,
                ),
            ))

        element_ratings_list = []
        for er in parsed.get("element_ratings", []):
            rating = er.get("condition_state") or er.get("rating")
            element_ratings_list.append(ElementRating(
                element_name=er.get("element") or "Unknown",
                element_rating=NbiScore(score=int(rating) if rating is not None else None),
                condition_observation=ConditionObservation(
                    location=er.get("location") or "General",
                    observation=er.get("observation") or "",
                    reference_images=er.get("reference_images") or [],
                ),
            ))

        # Validate inspection_type against allowed Literal values
        raw_type = parsed.get("inspection_type", "Unknown")
        valid_types = ["Routine", "In-Depth", "Special", "Other", "Unknown"]
        insp_type = next((t for t in valid_types if t.lower() == str(raw_type).lower()), "Other")

        result = BridgeInspectionExtraction(
            agency=parsed.get("agency") or "Unknown",
            bin=parsed.get("bin") or "Unknown",
            inspection_type=insp_type,
            inspection_date=parsed.get("inspection_date") or "Unknown",
            condition_ratings=condition_ratings_list,
            element_ratings=element_ratings_list,
        )
    except Exception as e:
        logfire.error(f"Failed to parse condition ratings: {e}\nRaw: {raw[:500]}")
        result = BridgeInspectionExtraction(
            agency="Unknown", bin="Unknown", inspection_type="Unknown",
            inspection_date="Unknown", condition_ratings=[], element_ratings=[],
        )

    if output_path:
        out_file = Path(output_path).parent / "condition_ratings.json"
        with open(out_file, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

    return result


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    before_sleep=lambda rs: logfire.info(f"Retrying QA generation in {rs.next_action.sleep}s..."),
)
async def generate_qa_pairs(
    pdf_path: str,
    image_paths: list[str],
    image_reference_mappings: list[ImageReferencePair],
    condition_ratings: BridgeInspectionExtraction,
    output_path: str = None,
    use_cache: bool = True,
) -> list[QAPair]:
    """Generate QA pairs from the report, mappings, and ratings."""
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "qa_pairs.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                if isinstance(data, dict) and "qa_pairs" in data:
                    data = data["qa_pairs"]
                if isinstance(data, list) and data:
                    return [QAPair.model_validate(item) for item in data]

    report_text = _extract_text_from_pdf(pdf_path)
    mappings_text = json.dumps([m.model_dump() for m in image_reference_mappings], indent=2)
    ratings_text = json.dumps(condition_ratings.model_dump(), indent=2)

    prompt = (
        f"{QA_GENERATION_INSTRUCTIONS}\n\n"
        "## Extracted Condition Ratings (PRIMARY SOURCE FOR VERBATIM ANSWERS):\n"
        f"{ratings_text}\n\n"
        "## Image Reference Mappings:\n"
        f"{mappings_text}\n\n"
        f"Inspection Report:\n{report_text}\n\n"
        "Return a JSON array of QA pairs. Each element:\n"
        '{"question": "...", "answer": "...", "reference_images": ["img1.png", "img2.png"], '
        '"condition_rating": N_or_null}\n'
        "Return ONLY the JSON array, no other text."
    )

    logfire.info(f"Running local QA generation with {len(image_reference_mappings)} mappings")
    raw = await _chat([{"role": "user", "content": prompt}], max_tokens=8192)

    try:
        parsed = _extract_json(raw)
        result = []
        for item in parsed:
            # condition_rating needs to be an NbiScore object, not a plain int
            cr = item.get("condition_rating")
            if cr is not None and not isinstance(cr, dict):
                item["condition_rating"] = NbiScore(score=int(cr) if cr else None)
            elif cr is None:
                item["condition_rating"] = None
            # Ensure reference_images is a list
            if "reference_images" not in item:
                item["reference_images"] = []
            result.append(QAPair.model_validate(item))
    except Exception as e:
        logfire.error(f"Failed to parse QA pairs: {e}\nRaw: {raw[:500]}")
        result = []

    if output_path:
        out_file = Path(output_path).parent / "qa_pairs.json"
        with open(out_file, "w") as f:
            json.dump([p.model_dump() for p in result], f, indent=2)

    return result


async def pdf_qa_gen(
    pdf_path: str,
    image_paths: list[str],
    output_path: str,
    use_cache: bool = True,
) -> PDFGroundedQAGeneratedResponse:
    """
    Main pipeline — local-model equivalent of gemini_pdf_qa_gen.pdf_qa_gen.

    Args:
        pdf_path: Local path to the PDF file
        image_paths: Local paths to extracted images
        output_path: Path to save combined results
        use_cache: Whether to use cached results
    """
    logfire.info("Starting local PDF QA generation pipeline")

    image_reference_mappings = await make_image_reference_mappings(
        pdf_path, image_paths, output_path, use_cache,
    )

    condition_ratings = await extract_condition_ratings(
        pdf_path, image_reference_mappings, output_path, use_cache,
    )

    if not image_reference_mappings or not condition_ratings.condition_ratings:
        logfire.warn("Empty mappings or ratings, retrying without cache")
        image_reference_mappings = await make_image_reference_mappings(
            pdf_path, image_paths, output_path, use_cache=False,
        )
        condition_ratings = await extract_condition_ratings(
            pdf_path, image_reference_mappings, output_path, use_cache=False,
        )

    qa_pairs = await generate_qa_pairs(
        pdf_path, image_paths, image_reference_mappings,
        condition_ratings, output_path, use_cache,
    )

    full_response = PDFGroundedQAGeneratedResponse(
        image_reference_mappings=image_reference_mappings,
        condition_ratings=condition_ratings,
        qa_pairs=qa_pairs,
    )

    # Save combined response
    combined_path = Path(output_path).parent / "full_response_qa_pairs.json"
    with open(combined_path, "w") as f:
        json.dump(full_response.model_dump(), f, indent=2)

    return full_response
