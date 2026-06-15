from src.rag_platform.application.evidence_constraint_service import (
    EvidenceConstraintService,
)


def test_constraint_guard_detects_missing_time_condition() -> None:
    result = EvidenceConstraintService().find_gap(
        question="下单后十分钟内取消，退款多久能到账？",
        context="已支付订单审核通过后，退款通常1-3个工作日到账。",
    )

    assert result is not None
    assert result.missing_constraints == ["10分钟"]


def test_constraint_guard_accepts_supported_time_condition() -> None:
    result = EvidenceConstraintService().find_gap(
        question="下单后十分钟内取消，退款多久能到账？",
        context="下单后10分钟内取消不影响一般退款时效。",
    )

    assert result is None


def test_constraint_guard_ignores_question_without_exact_constraint() -> None:
    result = EvidenceConstraintService().find_gap(
        question="退款多久能到账？",
        context="不同支付渠道的退款到账时间不同。",
    )

    assert result is None
