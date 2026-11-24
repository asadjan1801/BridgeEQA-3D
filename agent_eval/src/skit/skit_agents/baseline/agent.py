from skit.skit_agents.base_inspection_agent import InspectionAgent
from agents import Agent
from skit.skit_agents.qa_response import InspectionQuestionAnswerResponse
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from skit.skit_agents.agent_prompts import EMBODIED_FORMATTING_AGENT_PROMPT
from typing import Coroutine, Any
from agents import Runner, RunResult
import logfire


class BaselineInspectionAgent(InspectionAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = "Baseline Inspection Agent"
        # Create reformatting agent for consistent output
        self.reformatting_agent = Agent(
            name="Formatting Agent",
            instructions=RECOMMENDED_PROMPT_PREFIX + EMBODIED_FORMATTING_AGENT_PROMPT,
            model=self.model_name,
            output_type=InspectionQuestionAnswerResponse,
        )

    def async_inspect(self, messages: list) -> Coroutine[Any, Any, RunResult]:
        message_ref = messages

        async def baseline_inspect():
            assert self.context is not None
            with logfire.span("Baseline Inspection Agent"):
                messages = message_ref
                # Convert RunResponse to list if it is not already
                if isinstance(messages, RunResult):
                    messages = messages.to_input_list()

                # Run main agent without output_type (natural text output)
                messages = await Runner.run(
                    self.agent, messages, context=self.context, max_turns=20
                )
                messages_list = messages.to_input_list()

                # Reformat the final output using the reformatting agent
                messages = await Runner.run(
                    self.reformatting_agent, messages_list, context=self.context
                )
                return messages

        return baseline_inspect()
