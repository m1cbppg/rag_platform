from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from src.rag_platform.evaluation.models import (
    DatasetSplit,
    Difficulty,
    EvalCaseType,
    ExpectedAction,
    SourceDocumentType,
)


class CaseModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class NoAnswerSubtype(StrEnum):
    KNOWLEDGE_GAP = "KNOWLEDGE_GAP"
    MISSING_CONDITION = "MISSING_CONDITION"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


class CaseQuotaPlan(CaseModel):
    case_type_counts: dict[EvalCaseType, int]
    split_case_type_counts: dict[DatasetSplit, dict[EvalCaseType, int]]
    no_answer_subtype_counts: dict[NoAnswerSubtype, int]
    split_no_answer_subtype_counts: dict[
        DatasetSplit,
        dict[NoAnswerSubtype, int],
    ]
    split_topics: dict[DatasetSplit, list[str]]
    generation_pool_multiplier: float = Field(default=1.25, ge=1.0, le=3.0)
    generation_batch_size: int = Field(default=5, ge=1, le=5)

    @computed_field
    @property
    def total_case_count(self) -> int:
        return sum(self.case_type_counts.values())

    @computed_field
    @property
    def split_totals(self) -> dict[str, int]:
        return {
            split.value: sum(counts.values())
            for split, counts in self.split_case_type_counts.items()
        }

    @model_validator(mode="after")
    def validate_quotas(self) -> "CaseQuotaPlan":
        expected_splits = {
            DatasetSplit.DEVELOPMENT: 180,
            DatasetSplit.VALIDATION: 60,
            DatasetSplit.TEST: 60,
        }
        if self.total_case_count != 300:
            raise ValueError("评测题总数必须为300")

        for case_type, total in self.case_type_counts.items():
            split_total = sum(
                counts.get(case_type, 0)
                for counts in self.split_case_type_counts.values()
            )
            if split_total != total:
                raise ValueError(
                    f"{case_type.value} 分片配额合计应为{total}，"
                    f"实际为{split_total}"
                )

        for split, expected in expected_splits.items():
            actual = sum(self.split_case_type_counts.get(split, {}).values())
            if actual != expected:
                raise ValueError(
                    f"{split.value} 题目数量应为{expected}，实际为{actual}"
                )

        no_answer_total = self.case_type_counts.get(
            EvalCaseType.NO_ANSWER,
            0,
        )
        if sum(self.no_answer_subtype_counts.values()) != no_answer_total:
            raise ValueError("NO_ANSWER 子类型配额合计不正确")

        for subtype, total in self.no_answer_subtype_counts.items():
            split_total = sum(
                counts.get(subtype, 0)
                for counts in self.split_no_answer_subtype_counts.values()
            )
            if split_total != total:
                raise ValueError(
                    f"{subtype.value} 分片配额合计应为{total}，"
                    f"实际为{split_total}"
                )

        for split, subtype_counts in (
            self.split_no_answer_subtype_counts.items()
        ):
            expected = self.split_case_type_counts[split].get(
                EvalCaseType.NO_ANSWER,
                0,
            )
            if sum(subtype_counts.values()) != expected:
                raise ValueError(
                    f"{split.value} 的NO_ANSWER子类型配额不正确"
                )

        topic_owners: dict[str, DatasetSplit] = {}
        for split, topics in self.split_topics.items():
            for topic in topics:
                owner = topic_owners.get(topic)
                if owner is not None and owner != split:
                    raise ValueError(f"主题{topic}不能分配到多个数据分片")
                topic_owners[topic] = split
        return self


class CaseSourceFact(CaseModel):
    source_doc_code: str
    fact_key: str
    fact_text: str
    chunk_ids: list[int] = Field(min_length=1)


class CaseSourceDocument(CaseModel):
    source_doc_code: str
    mapped_doc_id: int
    doc_type: SourceDocumentType
    title: str
    topic: str
    version: str
    version_group: str | None = None
    required_identifiers: list[str] = Field(default_factory=list)
    facts: list[CaseSourceFact] = Field(min_length=1)
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class CaseSeed(CaseModel):
    seed_code: str
    case_type: EvalCaseType
    dataset_split: DatasetSplit
    expected_action: ExpectedAction
    source_doc_codes: list[str] = Field(default_factory=list)
    source_topics: list[str] = Field(default_factory=list)
    source_group: str
    facts: list[CaseSourceFact] = Field(default_factory=list)
    target_doc_types: list[SourceDocumentType] = Field(default_factory=list)
    required_identifier: str | None = None
    version_group: str | None = None
    no_answer_subtype: NoAnswerSubtype | None = None
    variant_index: int = Field(ge=1)


class GeneratedCaseText(CaseModel):
    seed_code: str
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    difficulty: Difficulty


class GeneratedCaseBatch(CaseModel):
    cases: list[GeneratedCaseText] = Field(min_length=1, max_length=5)


class CaseReviewResult(CaseModel):
    case_code: str
    answerable: bool
    expected_action_correct: bool
    reference_answer_supported: bool
    evidence_complete: bool
    ambiguity_score: float = Field(ge=0, le=1)
    difficulty: Difficulty
    semantic_duplicate: bool = False
    issues: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def passed(self) -> bool:
        return (
            self.expected_action_correct
            and self.reference_answer_supported
            and self.evidence_complete
            and self.ambiguity_score <= 0.15
            and not self.semantic_duplicate
        )

    @computed_field
    @property
    def score(self) -> float:
        values = [
            float(self.expected_action_correct),
            float(self.reference_answer_supported),
            float(self.evidence_complete),
            1.0 - self.ambiguity_score,
            float(not self.semantic_duplicate),
        ]
        return round(sum(values) / len(values), 4)


class CaseReviewBatch(CaseModel):
    reviews: list[CaseReviewResult] = Field(min_length=1, max_length=5)
