import json
from typing import List
from agents import Agent, Runner
from skit.skit_agents.file_utils import find_openai_file, upload_file_to_openai
from pathlib import Path
import asyncio
import logfire
from .pdf_data_schemas import (
    ImageReferencePair,
    BridgeInspectionExtraction,
    QAPair,
    PDFGroundedQAGeneratedResponse,
)
from openai.types import FileObject


# check and upload image to openai files api, then return the file id's
def check_and_upload_images_to_openai(image_dir: str) -> List[FileObject]:
    logfire.info(f"Starting image upload check for directory: {image_dir}")
    # get list of images, png, jpg, jpeg
    image_extensions = {".png", ".jpg", ".jpeg"}
    image_paths = [
        img
        for img in Path(image_dir).iterdir()
        if img.suffix.lower() in image_extensions
    ]
    logfire.info(f"Found {len(image_paths)} images to process")
    image_files = []
    for image in image_paths:
        image_path = str(image)  # Full path to maintain directory context
        logfire.info(f"Processing image: {image_path}")
        image_file = find_openai_file(image_path, purpose="user_data")
        if not image_file:
            logfire.info(f"Uploading new image to OpenAI: {image_path}")
            image_file = upload_file_to_openai(image_path)
        else:
            logfire.info(f"Image already exists in OpenAI: {image_path}")
        image_files.append(image_file)
    logfire.info(f"Successfully processed {len(image_files)} images")
    return image_files


def check_and_upload_report_to_openai(report_path: str) -> FileObject:
    logfire.info(f"Checking report upload for: {report_path}")
    report_file = find_openai_file(report_path, purpose="user_data")
    if not report_file:
        logfire.info(f"Uploading new report to OpenAI: {report_path}")
        report_file = upload_file_to_openai(report_path)
    else:
        logfire.info(f"Report already exists in OpenAI: {report_path}")
    logfire.info(f"Report file ID: {report_file}")
    return report_file


# one agent makes the mappings between the images and the reference images
async def make_image_reference_mappings(
    report_path: str,
    images_path: str,
    output_path: str = None,
    model_name: str = "gpt-5",
    use_cache: bool = True,
) -> List[ImageReferencePair]:
    logfire.info(f"Starting image reference mapping with model: {model_name}")

    # Check for cached results
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "image_reference_mappings.json"
        if cache_file.exists():
            logfire.info(f"Loading cached image reference mappings from {cache_file}")
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            return [ImageReferencePair.model_validate(item) for item in cached_data]
        else:
            logfire.info(f"No cache found at {cache_file}, generating new mappings")
    elif not use_cache:
        logfire.info("Cache disabled, generating new image reference mappings")
    mapping_agent = Agent(
        name="Image Reference Mapping Agent",
        model=model_name,
        instructions="You are a helpful assistant that makes the mappings between the images and the reference images",
        output_type=List[ImageReferencePair],
    )
    logfire.info("Created mapping agent")
    image_files = check_and_upload_images_to_openai(images_path)
    report_file = check_and_upload_report_to_openai(report_path)
    # make the prompt
    messages = []
    for image_file in image_files:
        image_name = image_file.filename
        image_name = Path(image_name).name
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "file_id": image_file.id},
                    {"type": "input_text", "text": f"Image name: {image_name}"},
                ],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "input_file", "file_id": report_file.id}],
        }
    )

    logfire.info(f"Running mapping agent with {len(image_files)} images")
    runner_result = await Runner.run(mapping_agent, messages)
    result = runner_result.final_output
    logfire.info(
        f"Mapping completed successfully, generated {len(result) if result else 0} pairs"
    )

    # Save individual agent results if output_path provided
    if output_path:
        output_file = Path(output_path).parent / "image_reference_mappings.json"
        with open(output_file, "w") as f:
            json.dump(
                [pair.model_dump() for pair in result] if result else [], f, indent=2
            )
        logfire.info(f"Saved image reference mappings to {output_file}")

    return result


# another agent generates extracts the condition ratings from the report
async def extract_condition_ratings(
    report_path: str,
    output_path: str = None,
    model_name: str = "gpt-5",
    use_cache: bool = True,
) -> BridgeInspectionExtraction:
    logfire.info(f"Starting condition ratings extraction with model: {model_name}")

    # Check for cached results
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "condition_ratings.json"
        if cache_file.exists():
            logfire.info(f"Loading cached condition ratings from {cache_file}")
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            return BridgeInspectionExtraction.model_validate(cached_data)
        else:
            logfire.info(f"No cache found at {cache_file}, extracting new ratings")
    elif not use_cache:
        logfire.info("Cache disabled, extracting new condition ratings")
    extraction_agent = Agent(
        name="Condition Ratings Extraction Agent",
        model=model_name,
        instructions="You are a helpful assistant that extracts the condition ratings from the report",
        output_type=BridgeInspectionExtraction,
    )
    logfire.info("Created extraction agent")
    report_file = check_and_upload_report_to_openai(report_path)
    messages = [
        {
            "role": "user",
            "content": [{"type": "input_file", "file_id": report_file.id}],
        }
    ]
    logfire.info("Running extraction agent")
    runner_result = await Runner.run(extraction_agent, messages)
    result = runner_result.final_output
    logfire.info("Extraction completed successfully")

    # Save individual agent results if output_path provided
    if output_path:
        output_file = Path(output_path).parent / "condition_ratings.json"
        with open(output_file, "w") as f:
            json.dump(result.model_dump() if result else {}, f, indent=2)
        logfire.info(f"Saved condition ratings to {output_file}")

    return result


# another agent generates the Q/A pairs for the condition ratings, with the understanding that a person attempting to answer the question would only be able to see a set of images and not the report itself.
async def generate_qa_pairs(
    report_path: str,
    images_path: str,
    image_reference_mappings: List[ImageReferencePair],
    output_path: str = None,
    model_name: str = "gpt-5",
    use_cache: bool = True,
) -> List[QAPair]:
    logfire.info(f"Starting QA pair generation with model: {model_name}")

    # Check for cached results
    if output_path and use_cache:
        cache_file = Path(output_path).parent / "qa_pairs.json"
        if cache_file.exists():
            logfire.info(f"Loading cached QA pairs from {cache_file}")
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            return [QAPair.model_validate(item) for item in cached_data]
        else:
            logfire.info(f"No cache found at {cache_file}, generating new QA pairs")
    elif not use_cache:
        logfire.info("Cache disabled, generating new QA pairs")
    qa_agent = Agent(
        name="QA Generation Agent",
        model=model_name,
        instructions="""
        Your task is to generate challenging Question and Answer (Q&A) pairs by translating visual or highly descriptive observations of bridge components and defects (as found in inspection notes, photographs, or derived scene graphs) into formal National Bridge Inventory (NBI) numerical ratings or status codes.
        Task: Generate a list of 10 distinct Question and Answer (Q&A) pairs that require the inference of NBI coding based solely on qualitative visual data.
        Critical Constraints for Q&A Generation
        1. Visual and Descriptive Grounding (Question): The Question must be framed using qualitative descriptions of physically observable defects, conditions, materials, or geometric problems (e.g., spalling, cracking, exposed steel, alignment issues, scour, deterioration of protective systems). The Question MUST NOT include any direct reference to photo numbers, figure numbers, or inspection report section identifiers.
        2. Condition State Context (Question): Do not incorporate Condition State (CS 1, CS 2, CS 3, or CS 4) terminology into the question to define the severity or extent of the observed defects (e.g., "widespread minor damage [CS2]," or "exposed reinforcing steel [CS4]").
        3. Required Answer Output (Answer): The Answer MUST be incorporate these parts:
            ◦ Part 1 (Rating): A formal NBI numerical rating (0-9) for the relevant component.
            ◦ Part 2 (Reasoning): A brief explanation detailing why that specific rating/code was assigned, citing the documented conditions, the synthesis required, and, crucially, explaining any counter-intuitive result (i.e., the "trick").
        4. Synthesis and "Tricky" Requirement: Each question must require the integration or reconciliation of multiple pieces of descriptive or structural information. Questions should frequently employ a leading or contradictory structure, where the described visual condition suggests a low rating, but the reported NBI code is higher, or vice versa (e.g., severe defects described, yet the resulting NBI rating is high because the condition is considered "localized" or "non-structural" in the final rating context). All Questions and Answers must be substantiated from the source inspection report.
        """,
        output_type=List[QAPair],
    )
    logfire.info("Created QA generation agent")
    report_file = check_and_upload_report_to_openai(report_path)
    image_files = check_and_upload_images_to_openai(images_path)
    image_reference_mappings_json = []
    for mapping in image_reference_mappings:
        image_reference_mappings_json.append(mapping.model_dump_json())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": report_file.id},
                {
                    "type": "input_text",
                    "text": "Ensure references to images align with reference image names in the Image reference mappings: "
                    + str(image_reference_mappings_json),
                },
            ]
            + [
                {"type": "input_image", "file_id": image_file.id}
                for image_file in image_files
            ],
        }
    ]
    logfire.info(f"Running QA generation agent with {len(image_files)} images")
    runner_result = await Runner.run(qa_agent, messages)
    result = runner_result.final_output
    logfire.info(
        f"QA generation completed, created {len(result) if result else 0} Q/A pairs"
    )

    # Save individual agent results if output_path provided
    if output_path:
        output_file = Path(output_path).parent / "qa_pairs.json"
        with open(output_file, "w") as f:
            json.dump(
                [pair.model_dump() for pair in result] if result else [], f, indent=2
            )
        logfire.info(f"Saved QA pairs to {output_file}")

    return result


async def pdf_qa_gen(
    report_path: str,
    images_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-pro",
    use_cache: bool = True,
) -> PDFGroundedQAGeneratedResponse:
    logfire.info("Starting PDF QA generation pipeline")
    logfire.info(f"Report path: {report_path}")
    logfire.info(f"Images path: {images_path}")
    logfire.info(f"Output path: {output_path}")
    logfire.info(f"Model: {model_name}")
    logfire.info(f"Cache enabled: {use_cache}")

    async def generate_pdf_qa_gen():
        logfire.info("Running all three agents in parallel")

        # Run all three agents concurrently
        results = await asyncio.gather(
            make_image_reference_mappings(
                report_path, images_path, output_path, model_name, use_cache
            ),
            extract_condition_ratings(report_path, output_path, model_name, use_cache),
        )

        image_reference_mappings, condition_ratings = results
        qa_pairs = await generate_qa_pairs(
            report_path,
            images_path,
            image_reference_mappings,
            output_path,
            model_name,
            use_cache,
        )

        logfire.info("All agents completed, creating PDFQAGen object")
        return PDFGroundedQAGeneratedResponse(
            image_reference_mappings=image_reference_mappings,
            condition_ratings=condition_ratings,
            qa_pairs=qa_pairs,
        )

    pdf_qa_gen = await generate_pdf_qa_gen()
    logfire.info(f"Writing combined results to {"full_response_"+output_path}")
    with open("full_response_"+output_path, "w") as f:
        json.dump(pdf_qa_gen.model_dump(), f, indent=2)
    logfire.info("PDF QA generation pipeline completed successfully")

    logfire.warning(
        "QA pairs need to be updated to ensure the answers align with the inspection report interms of referenced image and reasoning."
    )
    return pdf_qa_gen
