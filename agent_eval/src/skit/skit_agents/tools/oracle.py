from agents import Agent, Runner, RunContextWrapper, function_tool
from skit.skit_agents.scene_context import SceneContext
from skit.skit_agents.scene_graph_schema import SceneGraph, Node
from skit.skit_agents.agent_prompts import ORACLE_AGENT_PROMPT
import json
import logfire
from skit.files import CloudFileStorage
import asyncio

@function_tool
async def oracle_tool(wrapper: RunContextWrapper[SceneContext], question: str) -> str:
    """Ask a human user simple questions to calibrate your severity assessment of visible damage or deterioration.

    The oracle is a human user (not a technical expert) who can see the image at your current node position and provide brief, simple answers. Use this tool to validate your perception of damage severity before assigning condition ratings.

    **When to Use**:
    - Before finalizing a condition rating
    - To calibrate whether deterioration is "minor" vs "moderate" vs "severe"
    - To validate if damage affects structural integrity

    **Good Questions** (brief, binary/simple):
    - "Does this rust look severe?"
    - "Is the deterioration extensive?"
    - "Would you call this structural damage?"
    - "Does this seem like major damage?"
    - "Is the cracking significant?"

    **Avoid**:
    - Asking for numerical ratings ("What rating would you give?")
    - Technical questions requiring expertise
    - Asking for the answer directly

    The oracle responds with 1-4 words typically (e.g., "Not particularly", "Fairly heavy", "Yes, appears to be").

    Args:
        question: A brief, simple question about severity, extent, or nature of visible conditions.

    Returns:
        The oracle's brief response with remaining call count.
    """

    context = wrapper.context
    reference_answer = context.reference_answer
    reference_condition_rating = context.reference_condition_rating
    reference_question = context.question
    logfire.warn(f"Oracle tool called with reference answer: {reference_answer}, reference condition rating: {reference_condition_rating}, reference question: {reference_question}")

    if context.scene_graph is None:
        scene_graph_path = context.project_path / "scene_graph.json"

        with open(scene_graph_path, "r") as f:
            scene_graph = json.load(f)
        context.scene_graph = SceneGraph.model_validate(scene_graph)
    scene_graph = context.scene_graph

    current_node: Node = scene_graph.nodes[context.current_position]

    logfire.info(f"Oracle tool called with agent position: {context.current_position}")
    logfire.info(f"Oracle tool called with current node: {current_node}")
    logfire.info(f"Oracle was asked the question: {question}")

    # node image data is in the context.project_path / "images" folder
    node_image_path = context.project_path / "images" / current_node.image_name
    if not node_image_path.exists():
        node_image_path = current_node.image_name  # fallback to just the image name if not found
        
    cfs = CloudFileStorage()
    # Get the download URL for the image
    cloud_response = cfs.get_download_url(node_image_path)
    image_url = cloud_response.download_url

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_url},
                {
                    "type": "input_text",
                    "text": f"A user asked an agent the question {reference_question}. In an attempt to answer the users question the agent is looking at this image, and is asking you the question: "
                    + question,
                },
            ],
        }
    ]

    oracle = Agent(
        name="Oracle",
        instructions=ORACLE_AGENT_PROMPT,
        model=context.tool_model_name,
    )

    if context.oracle_calls > context.max_oracle_calls:
        return "Oracle has reached the maximum number of calls."

    context.oracle_calls += 1
    response = await Runner.run(oracle, messages)
    return f"Oracle Response: {response.final_output}, Remaining oracle calls: {context.max_oracle_calls - context.oracle_calls}. As a reminder, you are currently positioned at node: {current_node.image_name}, containing the following information: {current_node}"
