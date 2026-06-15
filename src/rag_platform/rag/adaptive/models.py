from dataclasses import asdict, dataclass, field
from enum import StrEnum


ADAPTIVE_RETRIEVAL_POLICY_VERSION = "v1"


class RetrievalQualityLevel(StrEnum):
    GOOD = "GOOD"
    WEAK = "WEAK"
    POOR = "POOR"


class RetryStrategy(StrEnum):
    NONE = "NONE"
    QUERY_REWRITE = "QUERY_REWRITE"
    FORCE_BM25 = "FORCE_BM25"
    RELAX_FILTER = "RELAX_FILTER"


@dataclass(frozen=True)
class RetrievalQualityFeatures:
    candidate_count: int
    distinct_document_count: int
    channel_overlap_at_10: float
    rerank_top1: float
    rerank_top3_mean: float
    rerank_margin: float
    target_type_coverage: float
    exact_terms: list[str] = field(default_factory=list)
    exact_term_coverage: float = 1.0
    distinct_version_count: int = 0
    comparison_intent: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalQualityDecision:
    level: RetrievalQualityLevel
    score: float
    retry_strategy: RetryStrategy
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "quality": self.level.value,
            "score": self.score,
            "retry_strategy": self.retry_strategy.value,
            "need_rewrite": self.retry_strategy != RetryStrategy.NONE,
            "reasons": list(self.reasons),
            "reason": "；".join(self.reasons),
        }


@dataclass(frozen=True)
class QueryRewriteResult:
    rewritten_query: str
    expanded_queries: list[str] = field(default_factory=list)
    reason: str = ""
    fallback_used: bool = False

    @property
    def all_queries(self) -> list[str]:
        values = [self.rewritten_query, *self.expanded_queries]
        return list(
            dict.fromkeys(
                value.strip()
                for value in values
                if value and value.strip()
            )
        )[:3]


@dataclass(frozen=True)
class DecomposedSubQuery:
    sub_query_id: str
    question: str
    target_doc_types: list[str] = field(default_factory=list)
    depends_on_sub_query_id: str | None = None
    is_template: bool = False

    def to_dict(self) -> dict:
        result = {
            "sub_query_id": self.sub_query_id,
            "question": self.question,
            "target_doc_types": list(self.target_doc_types),
        }
        if self.depends_on_sub_query_id is not None:
            result["depends_on_sub_query_id"] = (
                self.depends_on_sub_query_id
            )
        if self.is_template:
            result["is_template"] = True
        return result


@dataclass(frozen=True)
class QueryDecompositionResult:
    requires_decomposition: bool
    sub_queries: list[DecomposedSubQuery] = field(
        default_factory=list
    )
    decomposition_type: str = "NONE"
    benefit_score: float = 0.0
    reason: str = ""
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "requires_decomposition": self.requires_decomposition,
            "decomposition_type": self.decomposition_type,
            "benefit_score": self.benefit_score,
            "sub_queries": [
                item.to_dict() for item in self.sub_queries
            ],
            "reason": self.reason,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class IntermediateFactResult:
    success: bool
    intermediate_fact: str = ""
    evidence_quote: str = ""
    supporting_chunk_id: int | None = None
    confidence: float = 0.0
    reason: str = ""
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
