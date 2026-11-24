from skit.skit_agents.base_inspection_agent import InspectionAgent
import asyncio
from pathlib import Path
import logfire
from skit.skit_agents.qa_response import (
    InspectionQuestionAnswerResponse,
)
import json
import os
from skit.files import CloudFileStorage
from agents import RunResult
from skit.skit_agents.scene_context import SceneContext
from typing import Dict
from skit.pipelines.pdf_data_schemas import QAPair
from skit.skit_agents.scene_graph_schema import SceneGraph
from skit.experiment_tracker import ExperimentTracker, AgentConfigurations, ExperimentResults
from skit.agent_configurations import get_agents_for_experiments, AgentName, AgentMessages, AGENT_CONFIGS
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Global CloudFileStorage singleton
_cloud_storage_instance = None

def get_cloud_storage() -> CloudFileStorage:
    """Get or create the global CloudFileStorage instance.

    This singleton pattern ensures we only create one S3 client and disk cache
    for the entire evaluation run, improving performance and cache hit rates.
    """
    global _cloud_storage_instance
    if _cloud_storage_instance is None:
        _cloud_storage_instance = CloudFileStorage()
    return _cloud_storage_instance

def create_node_and_image_url_mapping(scene_dir: str, scene_graph: SceneGraph) -> Dict[int, str]:
    """Create a mapping from node index to image URL."""
    image_dir = scene_dir + "/images"
    image_dir = Path(image_dir)
    node_to_image_mapping = {}
    cfs = get_cloud_storage()
    for idx, node in enumerate(scene_graph.nodes):
        image_file_path = image_dir / Path(node.image_name)
        node_to_image_mapping[idx] =  cfs.get_download_url(image_file_path).download_url
    return node_to_image_mapping


def convert_node_references_to_image_paths(
    reference_images: list[str] | None,
    scene_graph: SceneGraph,
    scene_dir: Path
) -> list[str] | None:
    """
    Convert node references (e.g., "node 21", "Node 11") to actual image filenames.

    The agent may return references in various formats:
    - "node 21" or "Node 21" (node index)
    - "Abutment 1.png" (descriptive name that doesn't match actual filename)
    - Actual image hash filenames (already correct)

    This function:
    1. Checks if the reference is already a valid image filename (hash format)
    2. Tries to parse node indices and convert to image paths
    3. Falls back to trying to match against actual image names

    Args:
        reference_images: List of image references from agent output
        scene_graph: The scene graph containing node-to-image mappings
        scene_dir: Path to the scene directory

    Returns:
        List of actual image filenames (just the basename, not full path)
    """
    if not reference_images:
        return None

    converted_images = []
    image_dir = scene_dir / "images"

    # Create mapping of node index to image filename (basename only)
    node_to_filename = {}
    for idx, node in enumerate(scene_graph.nodes):
        # Extract just the filename from the full path
        filename = Path(node.image_name).name
        node_to_filename[idx] = filename

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

        # Check if it matches any actual image file in the directory
        if image_dir.exists():
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    if image_path.name == ref_str or image_path.stem in ref_str:
                        converted_images.append(image_path.name)
                        break

    return converted_images if converted_images else None

def prepare_images_for_prompt(scene_dir: str) -> list[dict]:
    image_dir = scene_dir + "/images"
    scene_dir = Path(scene_dir)
    cfs = get_cloud_storage()

    images = []
    for image_file_path in Path(image_dir).iterdir():
        if image_file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # logfire.info(f"Image file path: {image_file_path}")
        image_file_response = cfs.get_download_url(image_file_path)
        # logfire.info(f"Image file response: {image_file_response}")
        images.append({"type": "input_image", "image_url": image_file_response.download_url})
        images.append(
            {"type": "input_text", "text": f"Image name: {Path(image_file_response.local_file_path).name}"}
        )
    return images


def create_agent_configuration(agent_name: str, model_name: str) -> AgentConfigurations:
    """Convert agent name from AGENT_CONFIGS to AgentConfigurations model.

    Args:
        agent_name: Name of the agent from AGENT_CONFIGS (e.g., "embodied_sg_oracle_both")
        model_name: LLM model name (e.g., "litellm/openrouter/google/gemini-2.5-flash")

    Returns:
        AgentConfigurations instance ready for ExperimentTracker
    """
    config = AGENT_CONFIGS[agent_name]

    # Extract tool names from Tool objects
    tool_names = [tool.name for tool in config.tools]

    # Create descriptive instructions based on config
    instructions = f"{config.architecture} agent with context_type={config.context_type}"

    return AgentConfigurations(
        agent_llm=model_name,
        agent_name=agent_name,
        agent_instructions=instructions,
        agent_tools=tool_names
    )


async def parallel_agent_run(agents: list[InspectionAgent], messages: list[AgentMessages], agent_names: list[AgentName])-> list[RunResult]:

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=lambda retry_state: logfire.info(f"Agent call rate limited, retrying in {retry_state.next_action.sleep} seconds...")
    )
    async def run_single_agent(agent: InspectionAgent, agent_messages: AgentMessages, agent_name: AgentName):
        try:
            result = await agent.async_inspect(agent_messages)
            return {"success": True, "result": result, "agent_name": agent_name}
        except Exception as e:
            logfire.error(f"Error running agent {agent_name}: {e}", _exc_info=True)
            # Re-raise to let tenacity handle retries
            raise

    # Use gather with return_exceptions=True to handle final failures gracefully
    tasks = [run_single_agent(agent, agent_messages, agent_name)
             for agent, agent_messages, agent_name in zip(agents, messages, agent_names)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error dictionaries
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "success": False,
                "error": str(result),
                "agent_name": agent_names[i]
            })
        else:
            final_results.append(result)

    return final_results


async def process_scene(scene_dir: Path, model_name: str, experiment_tracker: ExperimentTracker):

    qa_file = scene_dir / "qa_pairs.json"
    scene_dir = scene_dir

    qa_pairs = [QAPair.model_validate(qa) for qa in json.load(open(qa_file))]
    agent_results = {}
    questions = []
    reference_answers = []
    condition_ratings = []
    logfire.info(f"Processing scene {scene_dir} with {len(qa_pairs)} QA pairs")
    for qa in qa_pairs:
        question = qa.question
        reference_answer = qa.answer
        condition_rating = qa.condition_rating

        # check if condition rating is None or not. If condition rating is not None append to the question to directly provide a condition rating in the response.
        if condition_rating is None:
            question = f"{question}. A numerical value for the condition rating is optional."
            continue
        else:
            question = f"{question}. A numerical value for the condition rating is required."


        # temp till experiment tracker is integrated
        questions.append(question)
        reference_answers.append(reference_answer)
        condition_ratings.append(condition_rating)
        image_context = prepare_images_for_prompt(str(scene_dir))

        scene_graph_path = scene_dir / "scene_graph.json"
        scene_graph = SceneGraph.model_validate_json(open(scene_graph_path).read())
        node_summary = "\n".join([f"Node index {i} maps to image: {node.image_name}" for i, node in enumerate(scene_graph.nodes)])
        scene_graph_context = [{"type": "input_text", "text": f"The current scene graph is: {scene_graph.model_dump_json()}\n {node_summary} "}]
        node_to_image_mapping = create_node_and_image_url_mapping(str(scene_dir), scene_graph)
        scene_context = SceneContext(
            websocket=None,
            project_id=scene_dir.name,
            session_id=scene_dir.name,
            current_position=0,
            project_path=scene_dir,
            scene_graph=scene_graph,
            oracle_calls=0,
            max_oracle_calls=2,
            image_context=image_context,
            tool_model_name=model_name, #using same model for tool and agent
            node_to_image_mapping=node_to_image_mapping,
        )

        inspection_agents, messages, inspection_agents_names = get_agents_for_experiments(scene_context, model_name, question, image_context, scene_graph_context)

        # Register agent configurations and check cache
        agent_configs_map = {}
        cached_results = []
        agents_to_run = []
        messages_to_run = []
        names_to_run = []

        for agent, agent_messages, agent_name in zip(inspection_agents, messages, inspection_agents_names):
            # Create and register agent configuration
            agent_config = create_agent_configuration(agent_name, model_name)
            agent_configs_map[agent_name] = experiment_tracker.create_or_get_agent(agent_config)

            # Check if experiment already exists in cache
            cached_experiment = experiment_tracker.get_experiment(agent_configs_map[agent_name], question)

            # Only use cached results if the experiment succeeded (not an error)
            if cached_experiment and not (cached_experiment.agent_answer and cached_experiment.agent_answer.startswith("ERROR:")):
                logfire.info(f"Found cached result for agent {agent_name}, skipping execution")
                # Add cached result in the same format as parallel_agent_run returns
                cached_results.append({
                    "success": cached_experiment.agent_answer is not None,
                    "agent_name": agent_name,
                    "cached": True,
                    "cached_experiment": cached_experiment
                })
            else:
                # Need to run this agent (either no cache or previous run failed)
                if cached_experiment:
                    logfire.info(f"Found failed cached result for agent {agent_name}, will retry execution")
                agents_to_run.append(agent)
                messages_to_run.append(agent_messages)
                names_to_run.append(agent_name)

        # Run only uncached agents
        if agents_to_run:
            logfire.info(f"Running {len(agents_to_run)} uncached agents out of {len(inspection_agents)} total")
            new_results = await parallel_agent_run(agents_to_run, messages_to_run, names_to_run)
        else:
            logfire.info("All agents have cached results, skipping execution")
            new_results = []

        # Combine cached and new results
        results = cached_results + new_results

        # Log successful results
        for result_dict in results:
            agent_name = result_dict["agent_name"]
            if result_dict.get("cached"):
                # Handle cached results
                cached_exp = result_dict["cached_experiment"]
                # logfire.info(f"Agent {agent_name} Result (CACHED): {cached_exp.agent_answer}")
                # logfire.info(f"Agent {agent_name} Condition Rating (CACHED) {cached_exp.agent_condition_rating} vs Reference Condition Rating {condition_rating}")
            elif result_dict["success"]:
                result = result_dict["result"]
                logfire.info(f"Agent {agent_name} Result: {result.final_output}")
                agent_rating = result.final_output.condition_rating.score if result.final_output.condition_rating else None
                logfire.info(f"Agent {agent_name} Condition Rating {agent_rating} vs Reference Condition Rating {condition_rating}")
            else:
                logfire.error(f"Agent {agent_name} failed: {result_dict['error']}")

        logfire.info(f"Question: {question}")
        logfire.info(f"Reference answer: {reference_answer}")
        logfire.info(f"Condition rating: {condition_rating}")

        # Process results - handle cached, successes, and failures
        for result_dict in results:
            agent_name = result_dict["agent_name"]

            if agent_results.get(agent_name) is None:
                agent_results[agent_name] = []

            if result_dict.get("cached"):
                # Handle cached results
                cached_exp = result_dict["cached_experiment"]
                agent_results[agent_name].append(
                    {
                        "scene_dir": scene_dir.name,
                        "question": question,
                        "reference_answer": reference_answer,
                        "agent_condition_rating": cached_exp.agent_condition_rating,
                        "reference_condition_rating": condition_rating.score if condition_rating else None,
                        "agent_answer": cached_exp.agent_answer,
                        "agent_reference_images": cached_exp.agent_reference_images,
                        "reference_images": qa.reference_images,
                        "error": None,
                        "cached": True,
                    }
                )
            elif result_dict["success"]:
                agent_final_output: InspectionQuestionAnswerResponse = result_dict["result"].final_output

                # Convert node references to actual image filenames
                converted_images = convert_node_references_to_image_paths(
                    agent_final_output.reference_images,
                    scene_graph,
                    scene_dir
                )

                # Save to experiment tracker (will update if previously failed)
                exp, updated = experiment_tracker.create_or_update_experiment(
                    agent_config=agent_configs_map[agent_name],
                    question=question,
                    experiment_datetime=str(datetime.now()),
                    scene_dir=scene_dir.name,
                    reference_answer=reference_answer,
                    agent_answer=agent_final_output.answer,
                    reference_images=qa.reference_images,
                    condition_rating=condition_rating.score if condition_rating else None,
                    agent_condition_rating=agent_final_output.condition_rating.score if agent_final_output and agent_final_output.condition_rating else None,
                    agent_reference_images=converted_images
                )
                if updated:
                    logfire.info(f"Updated previously failed experiment for agent {agent_name}")

                agent_results[agent_name].append(
                    {
                        "scene_dir": scene_dir.name,
                        "question": question,
                        "reference_answer": reference_answer,
                        "agent_condition_rating": agent_final_output.condition_rating.score if agent_final_output and agent_final_output.condition_rating else None,
                        "reference_condition_rating": condition_rating.score if condition_rating else None,
                        "agent_answer": agent_final_output.answer,
                        "agent_reference_images": converted_images,
                        "reference_images": qa.reference_images,
                        "error": None,
                        "cached": False,
                    }
                )
            else:
                # Record failure in results and experiment tracker (will update if previously failed)
                exp, updated = experiment_tracker.create_or_update_experiment(
                    agent_config=agent_configs_map[agent_name],
                    question=question,
                    experiment_datetime=str(datetime.now()),
                    scene_dir=scene_dir.name,
                    reference_answer=reference_answer,
                    agent_answer=f"ERROR: {result_dict['error']}",
                    reference_images=qa.reference_images,
                    condition_rating=condition_rating.score if condition_rating else None,
                    agent_condition_rating=None,
                    agent_reference_images=None
                )

                agent_results[agent_name].append(
                    {
                        "scene_dir": scene_dir.name,
                        "question": question,
                        "reference_answer": reference_answer,
                        "agent_condition_rating": None,
                        "reference_condition_rating": condition_rating.score if condition_rating else None,
                        "agent_answer": None,
                        "agent_reference_images": None,
                        "reference_images": qa.reference_images,
                        "error": result_dict["error"],
                        "cached": False,
                    }
                )

        # per agent results + aggregated dump to scene_dir. Best-effort: when the
        # scene corpus lives on a read-only mount these writes raise OSError. The
        # canonical store is experiments.db (OUTPUT_DIR); analysis reads from there.
        try:
            for agent_name, results in agent_results.items():
                with open(Path(scene_dir) / f"{agent_name}_agent_results.json", "w") as f:
                    json.dump(results, f, indent=2)
            with open(Path(scene_dir) / f"aggregated_agent_results.json", "w") as f:
                json.dump(agent_results, f, indent=2)
        except OSError as e:
            logfire.warn(f"Skipping per-scene JSON dump for {scene_dir.name}: {e}")

async def process_scene_with_semaphore(semaphore, output_dir, scene_dir, model_name, experiment_tracker):
    """Process a single scene with semaphore to limit concurrency"""
    async with semaphore:
        logfire.info(f"Starting evaluation for {scene_dir}")
        try:
            await process_scene(scene_dir, model_name, experiment_tracker)
            logfire.info(f"Completed evaluation for {scene_dir}")
        except Exception as e:
            print(f"Error evaluating agent for {scene_dir}: {e}")
            logfire.error(
                f"Error evaluating agent for {scene_dir}: {e}", _exc_info=True
            )


async def run_concurrent_evaluations(scene_dirs_path: Path, output_dir: str, n_concurrent=3, max_scenes=10, model_name="gpt-5-mini"):
    """Run N evaluations concurrently using asyncio"""
    scene_dirs = os.listdir(scene_dirs_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize experiment tracker with model-specific database
    db_path = f"{output_dir}/experiments.db"
    experiment_tracker = ExperimentTracker(db_url=f"sqlite:///{db_path}")
    logfire.info(f"Initialized experiment tracker with database: {db_path}")

    # remove any paths that are not directories
    scene_dirs = [
        scene_dirs_path / scene_dir
        for scene_dir in scene_dirs
        if os.path.isdir(scene_dirs_path / scene_dir)
    ]

    # remove scene dirs that don't have a qa_pairs.json file
    scene_dirs = [
        scene_dir for scene_dir in scene_dirs if (scene_dir / "qa_pairs.json").exists()
    ]

    # Limit to max_scenes
    scene_dirs = scene_dirs[:max_scenes]
    scene_dirs.reverse()
    logfire.info(f"Scene dirs with qa_pairs.json: {scene_dirs}")
    # scene_dirs = scene_dirs["processed_outputv2/viewreport (19)/qa_pairs.json"]

    # Create semaphore to limit concurrent evaluations
    semaphore = asyncio.Semaphore(n_concurrent)

    # Create tasks for all scenes
    tasks = [
        process_scene_with_semaphore(semaphore, output_dir, scene_dir, model_name, experiment_tracker)
        for scene_dir in scene_dirs
    ]

    # Run all tasks (semaphore will limit concurrency)
    await asyncio.gather(*tasks)

    logfire.info(
        f"Completed {len(scene_dirs)} scenes with {n_concurrent} concurrent workers"
    )

if __name__ == "__main__":
    # Configure model, concurrency and max scenes
    MODEL_NAME = "litellm/openrouter/google/gemini-2.5-flash-lite"  # Model to use for evaluation
    # MODEL_NAME = "gpt-5-mini-2025-08-07"
    # MODEL_NAME = "litellm/openrouter/google/gemini-2.5-pro"
    # MODEL_NAME = "litellm/openrouter/google/gemini-2.5-flash"  # Model to use for evaluation
    # MODEL_NAME = "litellm/openrouter/google/gemini-2.5-flash-lite-preview-09-2025" # flash light react agent keeps breaking.
    # MODEL_NAME = "litellm/openrouter/x-ai/grok-4-fast"
    N_CONCURRENT = (
        10  # Number of concurrent evaluations (adjust based on API rate limits)
    )
    MAX_SCENES = 200  # Maximum number of scenes to process
    scene_dirs_path = Path("pdf_processed_vermont_pages_greater_than_20")
    OUTPUT_DIR = f"{scene_dirs_path.name}_eval_{Path(MODEL_NAME).name}_cached_output"

    # Run with async concurrency
    asyncio.run(
        run_concurrent_evaluations(
            scene_dirs_path=scene_dirs_path, output_dir=OUTPUT_DIR, n_concurrent=N_CONCURRENT, max_scenes=MAX_SCENES, model_name=MODEL_NAME
        )
    )
