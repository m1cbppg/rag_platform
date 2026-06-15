from pydantic import BaseModel, ConfigDict, Field

from src.rag_platform.domain.answer_action import (
    AnswerAction,
    AnswerDecisionSource,
)


class ClarificationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_clarification: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    clarification_question: str | None = None
    missing_conditions: list[str] = Field(default_factory=list)


class AnswerabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerable: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)


class AnswerActionDecision(BaseModel):
    action: AnswerAction
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    clarification_question: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    decision_source: AnswerDecisionSource
