import pytest

from src.rag_platform.evaluation.failure_attribution import (
    AttributionCode,
    attribute_case,
)


def _result(**overrides) -> dict:
    result = {
        "case_code": "CASE_001",
        "case_type": "DIRECT",
        "difficulty": "EASY",
        "expected_action": "ANSWER",
        "actual_action": "ANSWER",
        "status": "SUCCESS",
        "action_correct": 1,
        "fact_coverage": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "judge_passed": 1,
        "faithfulness_score": 1.0,
        "answer_relevance_score": 1.0,
        "completeness_score": 1.0,
        "citation_entailment_score": 1.0,
        "conflict_handling_score": None,
        "error_message": None,
    }
    result.update(overrides)
    return result


def _evidence(chunk_id: int, fact_key: str) -> dict:
    return {
        "mapped_chunk_id": chunk_id,
        "fact_key": fact_key,
        "mapping_status": "MAPPED",
    }


def _hit(channel: str, chunk_id: int) -> dict:
    return {
        "channel": channel,
        "chunk_id": chunk_id,
        "rank_no": 1,
    }


def test_attributes_complete_retrieval_miss_before_answer_quality() -> None:
    attribution = attribute_case(
        case_result=_result(
            fact_coverage=0.0,
            judge_passed=0,
            completeness_score=0.0,
        ),
        hits=[],
        evidences=[_evidence(101, "fact_a")],
    )

    assert attribution.primary_code == AttributionCode.RETRIEVAL_MISS
    assert attribution.stage_fact_coverage == {
        "merged": 0.0,
        "rerank": 0.0,
        "final": 0.0,
    }
    assert AttributionCode.ANSWER_INCOMPLETE in attribution.secondary_codes


def test_attributes_partial_multi_fact_retrieval() -> None:
    attribution = attribute_case(
        case_result=_result(fact_coverage=0.5, judge_passed=0),
        hits=[
            _hit("HYBRID", 101),
            _hit("RERANK", 101),
            _hit("FINAL", 101),
        ],
        evidences=[
            _evidence(101, "fact_a"),
            _evidence(202, "fact_b"),
        ],
    )

    assert attribution.primary_code == AttributionCode.RETRIEVAL_PARTIAL
    assert attribution.stage_fact_coverage["merged"] == 0.5


@pytest.mark.parametrize(
    ("hits", "expected_code"),
    [
        (
            [_hit("HYBRID", 101)],
            AttributionCode.RERANK_DROPPED,
        ),
        (
            [_hit("HYBRID", 101), _hit("RERANK", 101)],
            AttributionCode.CONTEXT_DROPPED,
        ),
    ],
)
def test_attributes_stage_that_drops_gold(
    hits: list[dict],
    expected_code: AttributionCode,
) -> None:
    attribution = attribute_case(
        case_result=_result(fact_coverage=0.0, judge_passed=0),
        hits=hits,
        evidences=[_evidence(101, "fact_a")],
    )

    assert attribution.primary_code == expected_code


def test_attributes_false_refusal_after_gold_reaches_context() -> None:
    attribution = attribute_case(
        case_result=_result(
            actual_action="REFUSE",
            action_correct=0,
            judge_passed=0,
        ),
        hits=[
            _hit("HYBRID", 101),
            _hit("RERANK", 101),
            _hit("FINAL", 101),
        ],
        evidences=[_evidence(101, "fact_a")],
    )

    assert attribution.primary_code == AttributionCode.FALSE_REFUSAL


def test_attributes_false_answer_and_missing_clarification() -> None:
    false_answer = attribute_case(
        case_result=_result(
            expected_action="REFUSE",
            actual_action="ANSWER",
            action_correct=0,
            judge_passed=0,
        ),
        hits=[],
        evidences=[],
    )
    clarify = attribute_case(
        case_result=_result(
            expected_action="CLARIFY",
            actual_action="REFUSE",
            action_correct=0,
            judge_passed=0,
        ),
        hits=[],
        evidences=[],
    )

    assert false_answer.primary_code == AttributionCode.FALSE_ANSWER
    assert (
        clarify.primary_code
        == AttributionCode.CLARIFICATION_NOT_SUPPORTED
    )


@pytest.mark.parametrize(
    ("field", "case_type", "expected_code"),
    [
        (
            "citation_entailment_score",
            "DIRECT",
            AttributionCode.CITATION_FAILURE,
        ),
        (
            "completeness_score",
            "DIRECT",
            AttributionCode.ANSWER_INCOMPLETE,
        ),
        (
            "faithfulness_score",
            "DIRECT",
            AttributionCode.ANSWER_UNFAITHFUL,
        ),
        (
            "answer_relevance_score",
            "DIRECT",
            AttributionCode.ANSWER_IRRELEVANT,
        ),
        (
            "conflict_handling_score",
            "CONFLICT",
            AttributionCode.CONFLICT_HANDLING_FAILURE,
        ),
    ],
)
def test_attributes_answer_quality_failure(
    field: str,
    case_type: str,
    expected_code: AttributionCode,
) -> None:
    attribution = attribute_case(
        case_result=_result(
            case_type=case_type,
            judge_passed=0,
            **{field: 0.4},
        ),
        hits=[
            _hit("HYBRID", 101),
            _hit("RERANK", 101),
            _hit("FINAL", 101),
        ],
        evidences=[_evidence(101, "fact_a")],
    )

    assert attribution.primary_code == expected_code


def test_attributes_execution_error_and_passing_case() -> None:
    failed = attribute_case(
        case_result=_result(
            status="FAILED",
            actual_action="ERROR",
            error_message="连接失败",
        ),
        hits=[],
        evidences=[_evidence(101, "fact_a")],
    )
    passed = attribute_case(
        case_result=_result(),
        hits=[
            _hit("HYBRID", 101),
            _hit("RERANK", 101),
            _hit("FINAL", 101),
        ],
        evidences=[_evidence(101, "fact_a")],
    )

    assert failed.primary_code == AttributionCode.EXECUTION_ERROR
    assert passed.primary_code == AttributionCode.PASS
