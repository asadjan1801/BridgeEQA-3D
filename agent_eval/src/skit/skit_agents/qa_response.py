from pydantic import BaseModel, Field
from typing import Optional, List
from skit.pipelines.pdf_data_schemas import NbiScore

class InspectionQuestionAnswerResponse(BaseModel):
    answer: str = Field(..., description="Answer to a user's question")
    reference_images: Optional[List[str]] = Field(
        None, description="Optional reference images that are referenced in the answer"
    )
    condition_rating: Optional[NbiScore] = Field(
        None, description="Relevant NBI Condition rating for the answer when requested. This should be None if not applicable."
    )

class EmbodiedInspectionQuestionAnswerResponse(BaseModel):
    answer: str = Field(..., description="Answer to a user's question or what action must be taken next")
    reference_images: Optional[List[str]] = Field(
        None, description="Optional reference images that substantiate the answer to a user's question"
    )
    condition_rating: NbiScore = Field(
        None, description="Relevant NBI Condition rating for the final answer to the user's question"
    )
    action_required: bool = Field(
        ...,
        description="Set this to true if an action must be taken by the agent before a answer is ready.",
    )

class ReactInspectionQuestionAnswerResponse(BaseModel):
    answer: str = Field(..., description="Answer to a user's question")
    reference_images: Optional[List[str]] = Field(
        None, description="Optional reference images that are referenced in the answer"
    )
    condition_rating: NbiScore = Field(
        None, description="Relevant NBI Condition rating for the answer"
    )
    final_answer: bool = Field(
        ...,
        description="Set this to true if the answer is ready to be returned to the user and set to false if further actions must be taken by agents before a answer is ready.",
    )
    # guidance_from_think_agent_to_action_agent: Optional[str] = Field(
    #     None,
    #     description="Specific instructions from the Think Agent to guide the Action Agent's next steps, including what actions to take and what information to gather."
    # )
    # guidance_from_action_agent_to_feedback_agent: Optional[str] = Field(
    #     None,
    #     description="Contextual information from the Action Agent to help the Feedback Agent evaluate the action's effectiveness and provide informed feedback."
    # )
    # guidance_from_feedback_agent_to_think_agent: Optional[str] = Field(
    #     None,
    #     description="Evaluation feedback and recommendations from the Feedback Agent to inform the Think Agent's next planning iteration."
    # )

