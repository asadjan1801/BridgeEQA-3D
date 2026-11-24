from dataclasses import dataclass
from skit.skit_agents.baseline.agent import BaselineInspectionAgent
from skit.skit_agents.react.agent import ReactInspectionAgent
from skit.skit_agents.base_inspection_agent import InspectionAgent
from skit.skit_agents.embodied.agent import EmbodiedInspectionAgent
from skit.skit_agents.tools.map import scene_graph_interface_tool
from skit.skit_agents.tools.oracle import oracle_tool
from skit.skit_agents.scene_context import SceneContext
from typing import List
from agents import Tool


AgentName = str
AgentMessages = List[dict]


@dataclass
class AgentConfig:
    """Configuration for an agent variant in the evaluation."""
    architecture: str  # "baseline", "react", or "embodied"
    tools: List[Tool]  # List of tools available to the agent
    context_type: str  # "images", "sg_context", or "both"


# Agent registry with all 36 configurations (3 architectures × 4 tools × 3 contexts)
# Naming convention: {architecture}_{tools}_{context}
# - architecture: baseline, react, embodied
# - tools: no_tools, sg_tool, oracle_tool, sg_oracle
# - context: images, sg_context, both

AGENT_CONFIGS = {
    # ============================================================
    # BASELINE AGENT VARIANTS (12)
    # ============================================================
    # "baseline_no_tools_images": AgentConfig("baseline", [], "images"),
    # "baseline_no_tools_sg_context": AgentConfig("baseline", [], "sg_context"),
    # "baseline_no_tools_both": AgentConfig("baseline", [], "both"),

    # "baseline_sg_tool_images": AgentConfig("baseline", [scene_graph_interface_tool], "images"),
    # "baseline_sg_tool_sg_context": AgentConfig("baseline", [scene_graph_interface_tool], "sg_context"),
    # "baseline_sg_tool_both": AgentConfig("baseline", [scene_graph_interface_tool], "both"),

    # "baseline_oracle_tool_images": AgentConfig("baseline", [oracle_tool], "images"),
    # "baseline_oracle_tool_sg_context": AgentConfig("baseline", [oracle_tool], "sg_context"),
    # "baseline_oracle_tool_both": AgentConfig("baseline", [oracle_tool], "both"),

    # "baseline_sg_oracle_images": AgentConfig("baseline", [scene_graph_interface_tool, oracle_tool], "images"),
    # "baseline_sg_oracle_sg_context": AgentConfig("baseline", [scene_graph_interface_tool, oracle_tool], "sg_context"),
    # "baseline_sg_oracle_both": AgentConfig("baseline", [scene_graph_interface_tool, oracle_tool], "both"),

    # # ============================================================
    # # REACT AGENT VARIANTS (12)
    # # ============================================================
    # "react_no_tools_images": AgentConfig("react", [], "images"),
    # "react_no_tools_sg_context": AgentConfig("react", [], "sg_context"),
    # "react_no_tools_both": AgentConfig("react", [], "both"),

    # "react_sg_tool_images": AgentConfig("react", [scene_graph_interface_tool], "images"),
    # "react_sg_tool_sg_context": AgentConfig("react", [scene_graph_interface_tool], "sg_context"),
    # "react_sg_tool_both": AgentConfig("react", [scene_graph_interface_tool], "both"),

    # "react_oracle_tool_images": AgentConfig("react", [oracle_tool], "images"),
    # "react_oracle_tool_sg_context": AgentConfig("react", [oracle_tool], "sg_context"),
    # "react_oracle_tool_both": AgentConfig("react", [oracle_tool], "both"),

    # "react_sg_oracle_images": AgentConfig("react", [scene_graph_interface_tool, oracle_tool], "images"),
    # "react_sg_oracle_sg_context": AgentConfig("react", [scene_graph_interface_tool, oracle_tool], "sg_context"),
    # "react_sg_oracle_both": AgentConfig("react", [scene_graph_interface_tool, oracle_tool], "both"),

    # ============================================================
    # EMBODIED AGENT VARIANTS (12)
    # ============================================================
    "embodied_no_tools_images": AgentConfig("embodied", [], "images"), #baseline
    "embodied_no_tools_sg_context": AgentConfig("embodied", [], "sg_context"),
    "embodied_no_tools_both": AgentConfig("embodied", [], "both"),

    # "embodied_sg_tool_images": AgentConfig("embodied", [scene_graph_interface_tool], "images"),
    # "embodied_sg_tool_sg_context": AgentConfig("embodied", [scene_graph_interface_tool], "sg_context"),
    # "embodied_sg_tool_both": AgentConfig("embodied", [scene_graph_interface_tool], "both"),

    # "embodied_oracle_tool_images": AgentConfig("embodied", [oracle_tool], "images"), # oracle can't be used without sg tool
    # "embodied_oracle_tool_sg_context": AgentConfig("embodied", [oracle_tool], "sg_context"),
    # "embodied_oracle_tool_both": AgentConfig("embodied", [oracle_tool], "both"),

    # "embodied_sg_oracle_images": AgentConfig("embodied", [scene_graph_interface_tool, oracle_tool], "images"),
    "embodied_sg_oracle_sg_context": AgentConfig("embodied", [scene_graph_interface_tool, oracle_tool], "sg_context"),
    "embodied_sg_oracle_both": AgentConfig("embodied", [scene_graph_interface_tool, oracle_tool], "both"),
}


# Agent class mapping
AGENT_CLASS_MAP = {
    "baseline": BaselineInspectionAgent,
    "react": ReactInspectionAgent,
    "embodied": EmbodiedInspectionAgent,
}


# def _convert_to_openai_format(content_list: list) -> list:
#     """Convert custom content format to OpenAI standard format.

#     Converts:
#     - {"type": "input_text", "text": "..."} -> {"type": "input_text", "text": "..."}
#     - {"type": "input_image", "image_url": "..."} -> {"type": "image_url", "image_url": {"url": "..."}}
#     """
#     converted = []
#     for item in content_list:
#         if item["type"] == "input_text":
#             converted.append({"type": "input_text", "text": item["text"]})
#         elif item["type"] == "input_image":
#             # Handle both string URLs and dict formats
#             image_url = item["image_url"]
#             if isinstance(image_url, str):
#                 converted.append({"type": "image_url", "image_url": {"url": image_url}})
#             else:
#                 converted.append({"type": "image_url", "image_url": image_url})
#         else:
#             # Pass through unknown types unchanged
#             converted.append(item)
#     return converted


def create_agent_from_config(
    agent_name: str,
    scene_context: SceneContext,
    model_name: str,
    question: str,
    image_context: list,
    scene_graph_context: list
) -> tuple[InspectionAgent, AgentMessages]:
    """Create an agent and its messages from a configuration name.

    Args:
        agent_name: Name of the agent configuration (key in AGENT_CONFIGS)
        scene_context: Scene context object
        model_name: Name of the model to use
        question: The question/task for the agent
        image_context: Raw image data context
        scene_graph_context: Scene graph JSON context

    Returns:
        Tuple of (agent instance, messages list)

    Raises:
        KeyError: If agent_name is not in AGENT_CONFIGS
    """
    if agent_name not in AGENT_CONFIGS:
        raise KeyError(f"Unknown agent configuration: {agent_name}. Available: {list(AGENT_CONFIGS.keys())}")

    config = AGENT_CONFIGS[agent_name]

    # Select agent class based on architecture
    agent_class = AGENT_CLASS_MAP[config.architecture]

    # Create agent instance
    agent = agent_class(
        model_name=model_name,
        context=scene_context,
        tools=config.tools
    )

    # Build messages based on context type
    if config.context_type == "sg_context":
        # Scene graph context only
        messages = [
            {
                "role": "user",
                "content": scene_graph_context,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                ]
            }
        ]
    elif config.context_type == "images":
        # Image context only
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "image data"}] + image_context,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                ]
            }
        ]
    elif config.context_type == "both":
        # Both scene graph and image context
        messages = [
            {
                "role": "user",
                "content": scene_graph_context,
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "image data"}] + image_context,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                ]
            }
        ]
    else:
        raise ValueError(f"Unknown context_type: {config.context_type}. Must be 'images', 'sg_context', or 'both'")

    return agent, messages


def get_agents_for_experiments(
    scene_context: SceneContext,
    model_name: str,
    question: str,
    image_context: list,
    scene_graph_context: list
) -> tuple[List[InspectionAgent], List[AgentMessages], List[AgentName]]:
    """Create all agent variants for the experiment.

    Args:
        scene_context: Scene context object
        model_name: Name of the model to use
        question: The question/task for the agents
        image_context: Raw image data context
        scene_graph_context: Scene graph JSON context

    Returns:
        Tuple of (agents list, messages list, agent names list)
    """
    eval_agents = []
    eval_agents_messages = []
    eval_agents_names = []

    for agent_name in AGENT_CONFIGS.keys():
        agent, messages = create_agent_from_config(
            agent_name=agent_name,
            scene_context=scene_context,
            model_name=model_name,
            question=question,
            image_context=image_context,
            scene_graph_context=scene_graph_context
        )
        eval_agents.append(agent)
        eval_agents_messages.append(messages)
        eval_agents_names.append(agent_name)

    return eval_agents, eval_agents_messages, eval_agents_names
