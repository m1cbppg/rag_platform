from src.rag_platform.application.clarification_policy_service import (
    ClarificationPolicyService,
)


def test_policy_detects_missing_order_status_for_cancellation() -> None:
    result = ClarificationPolicyService().detect(
        question="我的订单怎么取消？"
    )

    assert result is not None
    assert result.policy_code == "cancel_order"
    assert result.missing_slot_codes == ["order_status"]
    assert "订单" in result.clarification_question
    assert "状态" in result.clarification_question


def test_policy_skips_when_required_slot_is_present() -> None:
    result = ClarificationPolicyService().detect(
        question="已发货订单怎么取消？"
    )

    assert result is None


def test_policy_recognizes_colloquial_order_status() -> None:
    result = ClarificationPolicyService().detect(
        question="我的订单已经发货了还能取消吗？"
    )

    assert result is None


def test_policy_does_not_intercept_rule_comparison_question() -> None:
    result = ClarificationPolicyService().detect(
        question="订单取消规则的新旧版本有什么不同？"
    )

    assert result is None


def test_cancel_policy_does_not_intercept_post_cancel_refund_question() -> None:
    result = ClarificationPolicyService().detect(
        question="我已经取消了订单，退款多久能到账？"
    )

    assert result is None


def test_policy_does_not_intercept_condition_summary_question() -> None:
    result = ClarificationPolicyService().detect(
        question="修改收货地址需要满足什么条件？"
    )

    assert result is None


def test_address_policy_recognizes_colloquial_unshipped_status() -> None:
    assert (
        ClarificationPolicyService().detect(
            question="我的订单还没发货，能改个地址吗？"
        )
        is None
    )


def test_policy_does_not_intercept_sop_knowledge_question() -> None:
    result = ClarificationPolicyService().detect(
        question="处理仅退款申请时，第一步需要做什么？"
    )

    assert result is None


def test_policy_detects_missing_damage_discovery_stage() -> None:
    result = ClarificationPolicyService().detect(
        question="包裹破损了，我现在该怎么处理？"
    )

    assert result is not None
    assert result.policy_code == "package_damage"
    assert result.missing_slot_codes == ["damage_discovery_stage"]


def test_coupon_policy_asks_refund_scope_and_coupon_status() -> None:
    result = ClarificationPolicyService().detect(
        question="订单退款后，优惠券会退回来吗？"
    )

    assert result is not None
    assert result.missing_slot_codes == [
        "refund_scope",
        "coupon_validity_status",
    ]
    assert "整单退款" in result.clarification_question
    assert "过期" in result.clarification_question


def test_refund_only_policy_asks_order_status_and_reason() -> None:
    result = ClarificationPolicyService().detect(
        question="我想申请仅退款，应该怎么处理？"
    )

    assert result is not None
    assert result.missing_slot_codes == [
        "refund_order_status",
        "refund_reason",
    ]
    assert "订单" in result.clarification_question
    assert "原因" in result.clarification_question


def test_policy_does_not_match_unrelated_question() -> None:
    result = ClarificationPolicyService().detect(
        question="退款一般多久到账？"
    )

    assert result is None
