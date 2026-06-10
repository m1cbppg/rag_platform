from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetStatus(StrEnum):
    DRAFT = "DRAFT"
    GENERATED = "GENERATED"
    REVIEWED = "REVIEWED"
    FROZEN = "FROZEN"
    ARCHIVED = "ARCHIVED"


class SourceDocumentType(StrEnum):
    FAQ = "FAQ"
    SOP = "SOP"
    RULE = "RULE"
    MANUAL = "MANUAL"


class EvalCaseType(StrEnum):
    DIRECT = "DIRECT"
    PARAPHRASE = "PARAPHRASE"
    EXACT = "EXACT"
    MULTI_CONDITION = "MULTI_CONDITION"
    MULTI_HOP = "MULTI_HOP"
    CONFLICT = "CONFLICT"
    NO_ANSWER = "NO_ANSWER"


class ExpectedAction(StrEnum):
    ANSWER = "ANSWER"
    REFUSE = "REFUSE"
    CLARIFY = "CLARIFY"


class ActualAction(StrEnum):
    ANSWER = "ANSWER"
    REFUSE = "REFUSE"
    CLARIFY = "CLARIFY"
    ERROR = "ERROR"


class Difficulty(StrEnum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class DatasetSplit(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    UNASSIGNED = "UNASSIGNED"


class ReviewStatus(StrEnum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    REJECTED = "REJECTED"


class MappingStatus(StrEnum):
    PENDING = "PENDING"
    MAPPED = "MAPPED"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING = "MISSING"


class EvalCaseStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class EvalRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EvalCaseResultStatus(StrEnum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class EvaluationModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class SourceDocumentSpec(EvaluationModel):
    source_doc_code: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    doc_type: SourceDocumentType
    topic: str = Field(min_length=1, max_length=100)
    version: str | None = Field(default=None, max_length=50)
    effective_from: date | None = None
    effective_to: date | None = None
    is_current: bool = True
    relative_file_path: str = Field(min_length=1, max_length=500)
    source_content_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    generation_spec: dict[str, Any] = Field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_score: float | None = Field(default=None, ge=0.0, le=1.0)
    review_reason: str | None = None
    mapped_doc_id: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_effective_dates(self) -> "SourceDocumentSpec":
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to 不能早于 effective_from")

        return self


class EvidenceSpec(EvaluationModel):
    source_doc_code: str = Field(min_length=1, max_length=64)
    evidence_quote: str = Field(min_length=1)
    fact_key: str = Field(min_length=1, max_length=100)
    relevance_grade: int = Field(default=1, ge=0, le=3)
    mapped_doc_id: int | None = Field(default=None, gt=0)
    mapped_chunk_id: int | None = Field(default=None, gt=0)
    mapping_status: MappingStatus = MappingStatus.PENDING
    mapping_reason: str | None = None

    @model_validator(mode="after")
    def validate_mapped_evidence(self) -> "EvidenceSpec":
        if self.mapping_status == MappingStatus.MAPPED:
            if self.mapped_doc_id is None or self.mapped_chunk_id is None:
                raise ValueError("MAPPED 证据必须同时包含 mapped_doc_id 和 mapped_chunk_id")

        return self


class GeneratedEvalCase(EvaluationModel):
    case_code: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1)
    normalized_question: str | None = None
    reference_answer: str | None = None
    case_type: EvalCaseType
    target_doc_types: list[SourceDocumentType] = Field(default_factory=list)
    expected_action: ExpectedAction = ExpectedAction.ANSWER
    difficulty: Difficulty = Difficulty.MEDIUM
    dataset_split: DatasetSplit = DatasetSplit.UNASSIGNED
    business_domain: str | None = Field(default=None, max_length=100)
    required_fact_count: int = Field(default=1, ge=0)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    evidences: list[EvidenceSpec] = Field(default_factory=list)

    @field_validator("reference_answer")
    @classmethod
    def normalize_reference_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_case_contract(self) -> "GeneratedEvalCase":
        if self.expected_action == ExpectedAction.ANSWER:
            if self.reference_answer is None:
                raise ValueError("expected_action=ANSWER 时必须提供 reference_answer")
            if not self.evidences:
                raise ValueError("expected_action=ANSWER 时必须提供标准证据")

        if self.case_type == EvalCaseType.NO_ANSWER:
            if self.expected_action not in {
                ExpectedAction.REFUSE,
                ExpectedAction.CLARIFY,
            }:
                raise ValueError(
                    "NO_ANSWER 题的 expected_action 必须是 REFUSE 或 CLARIFY"
                )
            if self.evidences:
                raise ValueError("NO_ANSWER 题不能包含标准证据")
            if self.required_fact_count != 0:
                raise ValueError("NO_ANSWER 题的 required_fact_count 必须是 0")

        if self.case_type == EvalCaseType.MULTI_HOP:
            fact_keys = {item.fact_key for item in self.evidences}

            if self.required_fact_count < 2:
                raise ValueError("MULTI_HOP 题的 required_fact_count 至少为 2")
            if len(fact_keys) < 2:
                raise ValueError("MULTI_HOP 题至少需要两个不同的 fact_key")
            if len(fact_keys) < self.required_fact_count:
                raise ValueError("标准证据中的不同 fact_key 数量少于 required_fact_count")

        return self


class ReviewedEvalCase(GeneratedEvalCase):
    review_status: ReviewStatus
    review_score: float | None = Field(default=None, ge=0.0, le=1.0)
    review_reason: str | None = None
    status: EvalCaseStatus = EvalCaseStatus.ACTIVE


class EvalRunConfig(EvaluationModel):
    run_code: str = Field(min_length=1, max_length=64)
    dataset_id: int = Field(gt=0)
    experiment_version: str = Field(min_length=1, max_length=30)
    experiment_name: str = Field(min_length=1, max_length=255)
    git_commit_sha: str | None = Field(default=None, max_length=64)
    retrieval_mode: str | None = Field(default=None, max_length=30)
    embedding_model: str | None = Field(default=None, max_length=100)
    rerank_model: str | None = Field(default=None, max_length=100)
    answer_model: str | None = Field(default=None, max_length=100)
    judge_model: str | None = Field(default=None, max_length=100)
    config: dict[str, Any]
    total_cases: int = Field(default=0, ge=0)

    @field_validator("config")
    @classmethod
    def validate_config_snapshot(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("实验必须保存非空配置快照")

        return value


class RetrievalMetricResult(EvaluationModel):
    recall_at_1: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_at_3: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    ndcg_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    ndcg_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    fact_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    action_correct: bool | None = None
    retrieval_rounds: int = Field(default=1, ge=1)


class JudgeScore(EvaluationModel):
    judge_provider: str = Field(min_length=1, max_length=50)
    judge_model: str = Field(min_length=1, max_length=100)
    judge_prompt_version: str = Field(min_length=1, max_length=30)
    faithfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    completeness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_entailment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    conflict_handling_score: float | None = Field(default=None, ge=0.0, le=1.0)
    refusal_correct: bool | None = None
    clarification_correct: bool | None = None
    passed: bool
    reason: dict[str, Any]
    raw_response: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = Field(default=None, ge=0)
