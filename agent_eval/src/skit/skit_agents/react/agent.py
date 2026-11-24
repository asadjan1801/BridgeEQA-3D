from skit.skit_agents.base_inspection_agent import InspectionAgent
from agents import Agent, Tool
from skit.skit_agents.qa_response import (
    InspectionQuestionAnswerResponse,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from typing import Coroutine
import logfire
from agents import Runner, RunResult
from skit.skit_agents.agent_prompts import (
    DEFAULT_INSPECTION_GUIDELINES_PROMPT,
    ORACLE_TOOL_GUIDANCE,
    SCENE_GRAPH_TOOL_GUIDANCE,
    REACT_FORMATTING_AGENT_PROMPT,
    REACT_THINK_AGENT_PROMPT,
    REACT_ACTION_AGENT_PROMPT,
    REACT_FEEDBACK_AGENT_PROMPT,
)
from typing import Any
class ReactInspectionAgent(InspectionAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = "React Inspection Agent"
        self.tools = kwargs.get("tools", [])
        self.agent = self._create_react_agent()

    def _create_react_agent(self):
        # Build base instructions WITHOUT tool-specific guidance (tools only for action agent)
        base_instructions = DEFAULT_INSPECTION_GUIDELINES_PROMPT

        # Build action agent instructions WITH tool-specific guidance
        action_instructions = self._build_react_instructions()

        self.reformatting_agent = Agent(
            name="Formatting Agent",
            instructions=RECOMMENDED_PROMPT_PREFIX + REACT_FORMATTING_AGENT_PROMPT,
            model=self.model_name,
            output_type=InspectionQuestionAnswerResponse,
        )

        tool_descriptions = ", ".join([tool.name + ": " + tool.description for tool in self.tools]) if self.tools else "No tools available"
        self.think_agent = Agent(
            name="Think Agent",
            instructions=RECOMMENDED_PROMPT_PREFIX + base_instructions + "\n\n" + REACT_THINK_AGENT_PROMPT.format(tool_descriptions=tool_descriptions),
            model=self.model_name,
            handoff_description="After thinking you must handoff to the action agent to take the action with the tools it has access to.",
        )

        self.action_agent = Agent(
            name="Action Agent",
            instructions=RECOMMENDED_PROMPT_PREFIX + action_instructions + "\n\n" + REACT_ACTION_AGENT_PROMPT,
            model=self.model_name,
            tools=self.tools,
            handoff_description="After taking the actions you must handoff to the feedback agent to give feedback on the action taken.",
        )

        self.feedback_agent = Agent(
            name="Feedback Agent",
            instructions=RECOMMENDED_PROMPT_PREFIX + base_instructions + "\n\n" + REACT_FEEDBACK_AGENT_PROMPT,
            model=self.model_name,
            handoffs=[self.think_agent],
            handoff_description="If the answer to the query requires further actions, you are responsible for giving feedback to the think agent to advise the action agent on the action to take with the tools it has access to.",
        )

        self.think_agent.handoffs = [self.action_agent]
        self.action_agent.handoffs = [self.feedback_agent]

        return self.think_agent

    def _build_react_instructions(self) -> str:
        """Build instructions with tool-specific guidance for React agents."""
        instructions = DEFAULT_INSPECTION_GUIDELINES_PROMPT

        # Check if oracle tool is available
        has_oracle = any(
            tool.name == "oracle_tool" if isinstance(tool, Tool) else False
            for tool in self.tools
        )

        # Check if scene_graph_interface_tool is available
        has_scene_graph = any(
            tool.name == "scene_graph_interface_tool" if isinstance(tool, Tool) else False
            for tool in self.tools
        )

        # Add tool-specific guidance
        if has_oracle:
            instructions += "\n\n" + ORACLE_TOOL_GUIDANCE

        if has_scene_graph:
            instructions += "\n\n" + SCENE_GRAPH_TOOL_GUIDANCE

        return instructions

    def async_inspect(self, messages: list) -> Coroutine[Any,Any,RunResult]:
        message_ref = messages

        async def react_inspect():
            assert self.context is not None
            with logfire.span("ReAct Agent (Not Embodied)"):
                messages = message_ref
                # convert RunResponse to list if it is not already
                if isinstance(messages, RunResult):
                    messages = messages.to_input_list()
                messages = await Runner.run(
                    self.think_agent, messages, context=self.context, max_turns=20
                )
                messages_list = messages.to_input_list()
                # reformat the final output using the reformatting agent
                messages = await Runner.run(
                    self.reformatting_agent, messages_list, context=self.context)
                return messages

        return react_inspect()
