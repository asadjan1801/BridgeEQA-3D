from abc import ABC
import asyncio
from typing import Optional, Coroutine, Any
from agents import Agent, Runner, RunResult, Tool
from skit.skit_agents.agent_prompts import (
    DEFAULT_INSPECTION_GUIDELINES_PROMPT,
    ORACLE_TOOL_GUIDANCE,
    SCENE_GRAPH_TOOL_GUIDANCE,
)
from skit.skit_agents.scene_context import SceneContext

class InspectionAgent(ABC):
    agent_name = "Inspection Agent"  # Default name, should be overridden

    def __init__(
        self,
        model_name: str = "gpt-5",
        inspection_guidelines_prompt: str = DEFAULT_INSPECTION_GUIDELINES_PROMPT,
        tools: list = [],
        context: Optional[SceneContext] = None,
    ):
        self.model_name = model_name
        self.context = context
        self.tools = tools

        # Build instructions dynamically based on available tools
        instructions = self._build_instructions(inspection_guidelines_prompt, tools)

        # Note: Baseline agent overrides async_inspect to use a reformatting agent
        # React and Embodied agents create their own agent instances and don't use this
        self.agent = Agent(
            name=self.agent_name,
            model=model_name,
            instructions=instructions,
            tools=tools,
        )

    def _build_instructions(self, base_prompt: str, tools: list) -> str:
        """Build instructions dynamically based on available tools for fair comparison."""
        instructions = base_prompt

        # Check if oracle tool is available
        has_oracle = any(
            tool.name == "oracle_tool" if isinstance(tool, Tool) else False
            for tool in tools
        )

        # Check if scene_graph_interface_tool is available
        has_scene_graph = any(
            tool.name == "scene_graph_interface_tool" if isinstance(tool, Tool) else False
            for tool in tools
        )

        # Add tool-specific guidance
        if has_oracle:
            instructions += "\n\n" + ORACLE_TOOL_GUIDANCE

        if has_scene_graph:
            instructions += "\n\n" + SCENE_GRAPH_TOOL_GUIDANCE

        return instructions

        # if self.context:
            # logfire.info(f"Using context: {self.context}")

    def get_agent(self):
        return self.agent

    def async_inspect(self, messages: list) -> Coroutine[Any,Any,RunResult]:
        return Runner.run(
            self.agent,
            messages,
            context=self.context,
            max_turns=20,
            # run_config=RunConfig(
            #     model_settings=ModelSettings(
            #         extra_args={"service_tier": "flex", "timeout": 9000}
            #     )
            # ),
        )

    def sync_inspect(self, messages: list) -> str:
        return asyncio.run(self.async_inspect(messages))
