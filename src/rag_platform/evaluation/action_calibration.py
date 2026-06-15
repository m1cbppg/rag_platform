from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.rag_platform.evaluation.models import (
    ExpectedAction,
    ReviewedEvalCase,
)


class ClarificationBranch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    condition_value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    chunk_ids: list[int] = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)


class ClarificationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    missing_condition_key: str = Field(min_length=1)
    missing_condition_label: str = Field(min_length=1)
    clarification_question: str = Field(min_length=1)
    acceptable_question_keywords: list[str] = Field(min_length=1)
    branches: list[ClarificationBranch] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_distinct_branches(self) -> "ClarificationContract":
        values = [branch.condition_value for branch in self.branches]
        if len(values) != len(set(values)):
            raise ValueError("branches.condition_value不能重复")
        return self

    @property
    def supporting_chunk_ids(self) -> set[int]:
        return {
            chunk_id
            for branch in self.branches
            for chunk_id in branch.chunk_ids
        }


def validate_action_calibration_cases(
    cases: list[ReviewedEvalCase],
    *,
    active_chunk_ids: set[int],
) -> list[str]:
    errors: list[str] = []
    for case in cases:
        if case.expected_action != ExpectedAction.CLARIFY:
            continue

        payload = case.generation_metadata.get(
            "clarification_contract"
        )
        if not payload:
            errors.append(
                f"{case.case_code}缺少clarification_contract"
            )
            continue

        try:
            contract = ClarificationContract.model_validate(payload)
        except ValidationError as exc:
            errors.append(
                f"{case.case_code}的clarification_contract不合法：{exc}"
            )
            continue

        inactive = sorted(
            contract.supporting_chunk_ids - active_chunk_ids
        )
        if inactive:
            joined = ",".join(str(chunk_id) for chunk_id in inactive)
            errors.append(
                f"{case.case_code}引用了非ACTIVE Chunk：{joined}"
            )
    return errors
