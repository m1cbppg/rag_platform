from datetime import date

import pytest
from pydantic import ValidationError

from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    EvalRunConfig,
    EvidenceSpec,
    ExpectedAction,
    GeneratedEvalCase,
    JudgeScore,
    MappingStatus,
    SourceDocumentSpec,
)


def make_evidence(
    *,
    fact_key: str = "refund_rule",
    mapping_status: MappingStatus = MappingStatus.MAPPED,
) -> EvidenceSpec:
    return EvidenceSpec(
        source_doc_code="RULE_REFUND_001",
        evidence_quote="未发货订单允许直接申请退款。",
        fact_key=fact_key,
        relevance_grade=3,
        mapped_doc_id=10,
        mapped_chunk_id=101,
        mapping_status=mapping_status,
    )


def test_source_document_rejects_invalid_effective_date_range() -> None:
    with pytest.raises(ValidationError):
        SourceDocumentSpec(
            source_doc_code="RULE_REFUND_001",
            title="退款规则",
            doc_type="RULE",
            topic="refund",
            effective_from=date(2026, 2, 1),
            effective_to=date(2026, 1, 1),
            relative_file_path="rules/refund.docx",
            source_content_sha256="a" * 64,
        )


def test_answer_case_requires_reference_answer() -> None:
    with pytest.raises(ValidationError):
        GeneratedEvalCase(
            case_code="CASE_001",
            question="未发货订单可以退款吗？",
            case_type=EvalCaseType.DIRECT,
            expected_action=ExpectedAction.ANSWER,
            dataset_split=DatasetSplit.DEVELOPMENT,
            evidences=[make_evidence()],
        )


def test_no_answer_case_rejects_evidence() -> None:
    with pytest.raises(ValidationError):
        GeneratedEvalCase(
            case_code="CASE_002",
            question="平台是否支持火星配送？",
            case_type=EvalCaseType.NO_ANSWER,
            expected_action=ExpectedAction.REFUSE,
            dataset_split=DatasetSplit.TEST,
            evidences=[make_evidence()],
        )


def test_multi_hop_case_requires_two_distinct_fact_keys() -> None:
    with pytest.raises(ValidationError):
        GeneratedEvalCase(
            case_code="CASE_003",
            question="物流丢失退款后优惠券是否退回？",
            reference_answer="需要同时依据物流退款和优惠券返还规则。",
            case_type=EvalCaseType.MULTI_HOP,
            expected_action=ExpectedAction.ANSWER,
            dataset_split=DatasetSplit.VALIDATION,
            required_fact_count=2,
            evidences=[
                make_evidence(fact_key="refund_rule"),
                make_evidence(fact_key="refund_rule"),
            ],
        )


def test_multi_hop_case_accepts_two_distinct_fact_keys() -> None:
    case = GeneratedEvalCase(
        case_code="CASE_004",
        question="物流丢失退款后优惠券是否退回？",
        reference_answer="需要同时依据物流退款和优惠券返还规则。",
        case_type=EvalCaseType.MULTI_HOP,
        expected_action=ExpectedAction.ANSWER,
        dataset_split=DatasetSplit.VALIDATION,
        required_fact_count=2,
        evidences=[
            make_evidence(fact_key="logistics_refund"),
            make_evidence(fact_key="coupon_return"),
        ],
    )

    assert case.required_fact_count == 2
    assert {item.fact_key for item in case.evidences} == {
        "logistics_refund",
        "coupon_return",
    }


def test_evidence_relevance_grade_must_be_between_zero_and_three() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpec(
            source_doc_code="RULE_REFUND_001",
            evidence_quote="退款规则。",
            fact_key="refund_rule",
            relevance_grade=4,
        )


def test_judge_scores_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        JudgeScore(
            judge_provider="dashscope",
            judge_model="qwen-plus",
            judge_prompt_version="v1",
            faithfulness_score=1.1,
            passed=False,
            reason={"message": "超出范围"},
        )


def test_eval_run_config_requires_reproducible_config_snapshot() -> None:
    with pytest.raises(ValidationError):
        EvalRunConfig(
            run_code="RUN_V0_001",
            dataset_id=1,
            experiment_version="V0",
            experiment_name="V0 基线",
            config={},
        )
