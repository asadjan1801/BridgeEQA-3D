from typing import Union
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class NbiScore(BaseModel):
    score: Union[int, None] = Field(
        None,
        description="""
        Leave empty if the score is not applicable.
        NBI condition rating (0-9):
        9 - Excellent condition
        8 - Very good condition
        7 - Good condition - some minor problems
        6 - Satisfactory condition - minor deterioration of structural elements (limited)
        5 - Fair condition - minor deterioration of structural elements (extensive)
        4 - Poor condition - deterioration significantly affects structural capacity
        3 - Serious condition - deterioration seriously affects structural capacity
        2 - Critical condition - bridge should be closed until repaired
        1 - Failing condition - bridge closed but repairable
        0 - Failed condition - bridge closed but beyond repair
        None - Not applicable or cannot be determined
        """,
    )
    # not_applicable: bool = Field(..., description="Whether the score is not applicable")
    # need_more_information: bool = Field(
    #     ..., description="Whether the score needs more information"
    # )


ComponentName = Literal[
    "Deck", "Superstructure", "Substructure", "Channel", "Culverts", "Approach"
]


class ConditionObservation(BaseModel):
    location: str = Field(
        ...,
        description="Specific location of the observation (e.g., 'Abutment 1', 'Pier 2') as found in report"
    )
    observation: str = Field(
        ...,
        description="EXACT text from the inspection report describing this condition. Must be a direct, verbatim quote from the report."
    )
    reference_images: Optional[List[str]] = Field(
        ...,
        description="Reference image names that show this condition, used to validate observations.",
    )


class ComponentRatings(BaseModel):
    component_prefix: str = Field(
        ...,
        description="Component prefix surrounding the component name as found in report to allow for disambiguation extraction of the component name. The combination of the prefix, suffix, and component name should recreate the naming of a component in any context given.",
    )
    component_suffix: str = Field(
        ...,
        description="Component suffix surrounding the component name as found in report to allow for disambiguation extraction of the component name. The combination of the prefix, suffix, and component name should recreate the naming of a component in any context given.",
    )
    component_name: ComponentName = Field(
        ..., description="Component name as found in report"
    )
    component_rating: NbiScore = Field(..., description="Component condition ratings")
    condition_observation: ConditionObservation = Field(
        ..., description="Observation of the component"
    )


class ElementRating(BaseModel):
    element_name: str = Field(..., description="Element name as found in report")
    element_rating: NbiScore = Field(..., description="Element condition rating")
    condition_observation: ConditionObservation = Field(
        ..., description="Observation of the element"
    )
