from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from skit.bridge_inspection_schemas import NbiScore, ComponentRatings, ElementRating


class ImageReferencePair(BaseModel):
    image_name: str = Field(..., description="Name of the image")
    reference_images: str = Field(
        ..., description="Name of the reference image in a document"
    )
    reference_image_page: int = Field(
        ..., description="Page number of the reference image in a document"
    )


class QAPair(BaseModel):
    question: str = Field(
        ...,
        description="""
        Inspection-focused question that:
        1. Uses observable condition descriptions (spalling, cracking, exposed rebar, etc.)
        2. Tests direct assessment skills (e.g., "What is the condition rating...")
        3. Explicitly requests ratings when applicable
        4. NEVER references photo/figure numbers or report sections
        5. NEVER uses Condition State codes (CS1, CS2, etc.) in the question
        """,
    )
    answer: str = Field(
        ...,
        description="""
        Nearly verbatim quote from the inspection report that answers the question.
        Must be an nearly exact match to text found in the source document, with only modification to ensure proper punctuation and grammar.
        For questions that ask for a condition rating, naturally incorporate the rating into the answer that matches the NBI condition_rating.
        """,
    )
    reference_images: list[str] = Field(
        description="Reference images that support the answer to the question. All answers must have multiple reference images that validate the answer."
    )
    condition_rating: Optional[NbiScore] = Field(
        None,
        description="NBI condition rating that is expected to be present in the answer by the question. Only used for rating questions."
    )

class BridgeInspectionExtraction(BaseModel):
    agency: Optional[str] = Field(
        None, description="Issuing agency as text (e.g., NYSDOT, VTrans, TxDOT)"
    )
    bin: Optional[str] = Field(
        None, description="Bridge Identification Number when present"
    )
    inspection_type: Optional[
        Literal["Routine", "In-Depth", "Special", "Other", "Unknown"]
    ] = Field(None, description="Type of inspection reported")
    inspection_date: Optional[str] = Field(
        None, description="Inspection date in ISO format YYYY-MM-DD"
    )
    condition_ratings: List[ComponentRatings] = Field(
        ...,
        description="Set of NBI/agency condition ratings for componentswith VERBATIM observations from the inspection report"
    )
    element_ratings: List[ElementRating] = Field(
        ...,
        description="Element-level condition state quantities with VERBATIM observations from the inspection report"
    )


class PDFGroundedQAGeneratedResponse(BaseModel):
    image_reference_mappings: List[ImageReferencePair] = Field(
        ...,
        description="Maps images to their references in the document"
    )
    condition_ratings: BridgeInspectionExtraction = Field(
        ...,
        description="Condition ratings with verbatim observations - PRIMARY SOURCE for Q/A answer text"
    )
    qa_pairs: List[QAPair] = Field(
        ...,
        description="""
        Q/A pairs where each answer is a VERBATIM quote from either:
        1. An observation field in condition_ratings (component or element)
        2. Direct text from the inspection report
        Answers must be EXACT matches to source text - no paraphrasing allowed.
        """
    )
