from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CitationMetricResult:
    precision: float | None
    recall: float | None
    true_positive_count: int
    cited_count: int
    gold_count: int


def citation_precision(
    cited_chunk_ids: Iterable[int],
    gold_chunk_ids: Iterable[int],
) -> float | None:
    return evaluate_citations(
        cited_chunk_ids=cited_chunk_ids,
        gold_chunk_ids=gold_chunk_ids,
    ).precision


def citation_recall(
    cited_chunk_ids: Iterable[int],
    gold_chunk_ids: Iterable[int],
) -> float | None:
    return evaluate_citations(
        cited_chunk_ids=cited_chunk_ids,
        gold_chunk_ids=gold_chunk_ids,
    ).recall


def evaluate_citations(
    *,
    cited_chunk_ids: Iterable[int],
    gold_chunk_ids: Iterable[int],
) -> CitationMetricResult:
    cited = set(cited_chunk_ids)
    gold = set(gold_chunk_ids)
    true_positive_count = len(cited & gold)
    if not gold:
        precision = None
        recall = None
    else:
        precision = (
            true_positive_count / len(cited)
            if cited
            else None
        )
        recall = true_positive_count / len(gold)
    return CitationMetricResult(
        precision=precision,
        recall=recall,
        true_positive_count=true_positive_count,
        cited_count=len(cited),
        gold_count=len(gold),
    )
