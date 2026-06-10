import math
from collections.abc import Iterable, Mapping


_RELEVANCE_GAINS = {
    0: 0.0,
    1: 1.0,
    2: 3.0,
    3: 7.0,
}


def recall_at_k(
    retrieved_ids: Iterable[int],
    relevant_ids: Iterable[int],
    k: int,
) -> float | None:
    _validate_k(k)
    gold = set(relevant_ids)
    if not gold:
        return None
    retrieved = set(_unique_in_order(retrieved_ids)[:k])
    return len(retrieved & gold) / len(gold)


def reciprocal_rank(
    retrieved_ids: Iterable[int],
    relevant_ids: Iterable[int],
) -> float | None:
    gold = set(relevant_ids)
    if not gold:
        return None
    for rank, chunk_id in enumerate(
        _unique_in_order(retrieved_ids),
        start=1,
    ):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Iterable[int],
    relevance_by_id: Mapping[int, int],
    k: int,
) -> float | None:
    _validate_k(k)
    if not relevance_by_id:
        return None
    gains = {
        chunk_id: _gain(grade)
        for chunk_id, grade in relevance_by_id.items()
    }
    retrieved = _unique_in_order(retrieved_ids)[:k]
    actual_dcg = _discounted_cumulative_gain(
        [gains.get(chunk_id, 0.0) for chunk_id in retrieved]
    )
    ideal_dcg = _discounted_cumulative_gain(
        sorted(gains.values(), reverse=True)[:k]
    )
    if ideal_dcg == 0:
        return None
    return actual_dcg / ideal_dcg


def fact_coverage(
    retrieved_ids: Iterable[int],
    fact_keys_by_chunk: Mapping[int, str | Iterable[str]],
) -> float | None:
    normalized = {
        chunk_id: _fact_key_set(fact_keys)
        for chunk_id, fact_keys in fact_keys_by_chunk.items()
    }
    required_facts = set().union(*normalized.values()) if normalized else set()
    if not required_facts:
        return None
    covered_facts: set[str] = set()
    for chunk_id in _unique_in_order(retrieved_ids):
        covered_facts.update(normalized.get(chunk_id, set()))
    return len(covered_facts & required_facts) / len(required_facts)


def _unique_in_order(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k必须大于0")


def _gain(grade: int) -> float:
    try:
        return _RELEVANCE_GAINS[grade]
    except KeyError as exc:
        raise ValueError("相关度等级必须是0、1、2或3") from exc


def _discounted_cumulative_gain(gains: Iterable[float]) -> float:
    return sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )


def _fact_key_set(value: str | Iterable[str]) -> set[str]:
    values = {value} if isinstance(value, str) else set(value)
    return {item for item in values if item}
