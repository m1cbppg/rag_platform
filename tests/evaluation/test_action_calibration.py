import pytest
from pydantic import ValidationError

from src.rag_platform.evaluation.action_calibration import (
    ClarificationContract,
    validate_action_calibration_cases,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    ExpectedAction,
    ReviewStatus,
    ReviewedEvalCase,
)


def _contract_payload() -> dict:
    return {
        "missing_condition_key": "order_status",
        "missing_condition_label": "订单状态",
        "clarification_question": "订单目前是待支付、待出库还是已发货？",
        "acceptable_question_keywords": ["订单状态", "待支付", "已发货"],
        "branches": [
            {
                "condition_value": "PENDING_PAYMENT",
                "label": "待支付",
                "chunk_ids": [50],
                "expected_outcome": "用户可以直接取消订单。",
            },
            {
                "condition_value": "SHIPPED",
                "label": "已发货",
                "chunk_ids": [54],
                "expected_outcome": "不能直接取消，需要走售后流程。",
            },
        ],
    }


def _case(metadata: dict) -> ReviewedEvalCase:
    return ReviewedEvalCase(
        case_code="ACTION_CLARIFY_001",
        question="我的订单怎么取消？",
        case_type=EvalCaseType.MULTI_CONDITION,
        expected_action=ExpectedAction.CLARIFY,
        dataset_split=DatasetSplit.DEVELOPMENT,
        required_fact_count=0,
        generation_metadata=metadata,
        evidences=[],
        review_status=ReviewStatus.PASSED,
        review_score=1.0,
        review_reason="人工按澄清契约审核通过",
    )


def test_clarification_contract_requires_two_distinct_branches() -> None:
    payload = _contract_payload()
    payload["branches"] = [payload["branches"][0]]

    with pytest.raises(ValidationError):
        ClarificationContract.model_validate(payload)


def test_action_calibration_rejects_missing_contract() -> None:
    errors = validate_action_calibration_cases(
        [_case(metadata={})],
        active_chunk_ids={50, 54},
    )

    assert errors == ["ACTION_CLARIFY_001缺少clarification_contract"]


def test_action_calibration_rejects_inactive_supporting_chunk() -> None:
    errors = validate_action_calibration_cases(
        [
            _case(
                metadata={
                    "clarification_contract": _contract_payload(),
                }
            )
        ],
        active_chunk_ids={50},
    )

    assert errors == ["ACTION_CLARIFY_001引用了非ACTIVE Chunk：54"]


def test_action_calibration_accepts_source_backed_contract() -> None:
    errors = validate_action_calibration_cases(
        [
            _case(
                metadata={
                    "clarification_contract": _contract_payload(),
                }
            )
        ],
        active_chunk_ids={50, 54},
    )

    assert errors == []
