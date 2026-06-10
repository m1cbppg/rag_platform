import math
import re
from dataclasses import dataclass

from src.rag_platform.evaluation.case_models import (
    CaseSeed,
    CaseSourceDocument,
)
from src.rag_platform.evaluation.models import (
    EvalCaseType,
    GeneratedEvalCase,
)


def normalize_question(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def validate_generated_case(
    *,
    case: GeneratedEvalCase,
    seed: CaseSeed,
    documents: dict[str, CaseSourceDocument],
) -> list[str]:
    errors: list[str] = []
    normalized_question = normalize_question(case.question)
    if not 6 <= len(normalized_question) <= 160:
        errors.append("QUESTION_LENGTH_INVALID")

    for code in seed.source_doc_codes:
        title = documents[code].title
        if normalize_question(title) in normalized_question:
            errors.append("QUESTION_LEAKS_DOCUMENT_TITLE")
            break

    if (
        seed.case_type == EvalCaseType.EXACT
        and seed.required_identifier
        and seed.required_identifier.casefold() not in case.question.casefold()
    ):
        errors.append("EXACT_QUESTION_MISSING_IDENTIFIER")

    expected_facts = {
        (fact.source_doc_code, fact.fact_key): fact.fact_text
        for fact in seed.facts
    }
    actual_facts = {
        (evidence.source_doc_code, evidence.fact_key): evidence.evidence_quote
        for evidence in case.evidences
    }
    if set(actual_facts) != set(expected_facts):
        errors.append("EVIDENCE_FACT_SET_MISMATCH")
    elif any(
        actual_facts[key] != fact_text
        for key, fact_text in expected_facts.items()
    ):
        errors.append("EVIDENCE_QUOTE_NOT_CANONICAL")

    if seed.case_type == EvalCaseType.MULTI_HOP:
        if len({item.fact_key for item in case.evidences}) < 2:
            errors.append("MULTI_HOP_FACT_COUNT_INVALID")
        if len({item.source_doc_code for item in case.evidences}) < 2:
            errors.append("MULTI_HOP_DOCUMENT_COUNT_INVALID")

    if seed.case_type == EvalCaseType.CONFLICT:
        if len({item.source_doc_code for item in case.evidences}) < 2:
            errors.append("CONFLICT_DOCUMENT_COUNT_INVALID")
        if not seed.version_group:
            errors.append("CONFLICT_VERSION_GROUP_MISSING")

    if case.reference_answer:
        answer = normalize_question(case.reference_answer)
        if len(answer) >= 8 and answer in normalized_question:
            errors.append("QUESTION_LEAKS_REFERENCE_ANSWER")
    return errors


@dataclass(frozen=True)
class SemanticDuplicateDecision:
    decision: str
    nearest_index: int | None
    similarity: float


class SemanticDeduplicator:
    def __init__(
        self,
        *,
        direct_threshold: float = 0.92,
        review_threshold: float = 0.85,
    ) -> None:
        self.direct_threshold = direct_threshold
        self.review_threshold = review_threshold

    def classify(
        self,
        vectors: list[list[float]],
    ) -> list[SemanticDuplicateDecision]:
        kept_indexes: list[int] = []
        results: list[SemanticDuplicateDecision] = []
        for index, vector in enumerate(vectors):
            if not kept_indexes:
                kept_indexes.append(index)
                results.append(
                    SemanticDuplicateDecision("KEEP", None, 0.0)
                )
                continue

            similarities = [
                (kept_index, _cosine(vector, vectors[kept_index]))
                for kept_index in kept_indexes
            ]
            nearest_index, similarity = max(
                similarities,
                key=lambda item: item[1],
            )
            if similarity >= self.direct_threshold:
                decision = "DUPLICATE"
            elif similarity >= self.review_threshold:
                decision = "REVIEW"
                kept_indexes.append(index)
            else:
                decision = "KEEP"
                kept_indexes.append(index)
            results.append(
                SemanticDuplicateDecision(
                    decision=decision,
                    nearest_index=nearest_index,
                    similarity=similarity,
                )
            )
        return results


def classify_grouped_semantic_duplicates(
    *,
    vectors: list[list[float]],
    group_keys: list[tuple[str, ...]],
    direct_threshold: float = 0.92,
    review_threshold: float = 0.85,
) -> list[SemanticDuplicateDecision]:
    if len(vectors) != len(group_keys):
        raise ValueError("向量数量与分组键数量不一致")
    indexes_by_group: dict[tuple[str, ...], list[int]] = {}
    for index, group_key in enumerate(group_keys):
        indexes_by_group.setdefault(group_key, []).append(index)

    results: list[SemanticDuplicateDecision | None] = [None] * len(vectors)
    for indexes in indexes_by_group.values():
        local_results = SemanticDeduplicator(
            direct_threshold=direct_threshold,
            review_threshold=review_threshold,
        ).classify([vectors[index] for index in indexes])
        for local_index, decision in enumerate(local_results):
            nearest_index = (
                indexes[decision.nearest_index]
                if decision.nearest_index is not None
                else None
            )
            results[indexes[local_index]] = SemanticDuplicateDecision(
                decision=decision.decision,
                nearest_index=nearest_index,
                similarity=decision.similarity,
            )
    return [result for result in results if result is not None]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("向量维度不一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )
