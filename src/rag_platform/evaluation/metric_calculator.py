from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.rag_platform.evaluation.action_metrics import action_correct
from src.rag_platform.evaluation.citation_metrics import evaluate_citations
from src.rag_platform.evaluation.models import (
    ActualAction,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
    RetrievalMetricResult,
)
from src.rag_platform.evaluation.retrieval_metrics import (
    fact_coverage,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class GoldAnnotations:
    relevance_by_chunk: dict[int, int]
    fact_keys_by_chunk: dict[int, set[str]]


def build_gold_annotations(
    evidences: Iterable[EvidenceSpec],
) -> GoldAnnotations:
    relevance_by_chunk: dict[int, int] = {}
    fact_keys_by_chunk: dict[int, set[str]] = {}
    for evidence in evidences:
        if (
            evidence.mapping_status != MappingStatus.MAPPED
            or evidence.mapped_chunk_id is None
        ):
            raise ValueError("Gold证据必须先完成Chunk映射")
        chunk_id = evidence.mapped_chunk_id
        relevance_by_chunk[chunk_id] = max(
            relevance_by_chunk.get(chunk_id, 0),
            evidence.relevance_grade,
        )
        fact_keys_by_chunk.setdefault(chunk_id, set()).add(
            evidence.fact_key
        )
    return GoldAnnotations(
        relevance_by_chunk=relevance_by_chunk,
        fact_keys_by_chunk=fact_keys_by_chunk,
    )


def calculate_case_metrics(
    *,
    retrieved_chunk_ids: Iterable[int],
    relevance_by_chunk: Mapping[int, int],
    fact_keys_by_chunk: Mapping[int, str | Iterable[str]],
    cited_chunk_ids: Iterable[int],
    expected_action: ExpectedAction | str,
    actual_action: ActualAction | str,
    retrieval_rounds: int = 1,
) -> RetrievalMetricResult:
    retrieved = list(retrieved_chunk_ids)
    gold_chunk_ids = list(relevance_by_chunk)
    citations = evaluate_citations(
        cited_chunk_ids=cited_chunk_ids,
        gold_chunk_ids=gold_chunk_ids,
    )
    return RetrievalMetricResult(
        recall_at_1=recall_at_k(retrieved, gold_chunk_ids, 1),
        recall_at_3=recall_at_k(retrieved, gold_chunk_ids, 3),
        recall_at_5=recall_at_k(retrieved, gold_chunk_ids, 5),
        recall_at_10=recall_at_k(retrieved, gold_chunk_ids, 10),
        reciprocal_rank=reciprocal_rank(retrieved, gold_chunk_ids),
        ndcg_at_5=ndcg_at_k(retrieved, relevance_by_chunk, 5),
        ndcg_at_10=ndcg_at_k(retrieved, relevance_by_chunk, 10),
        fact_coverage=fact_coverage(retrieved, fact_keys_by_chunk),
        citation_precision=citations.precision,
        citation_recall=citations.recall,
        action_correct=action_correct(expected_action, actual_action),
        retrieval_rounds=retrieval_rounds,
    )
