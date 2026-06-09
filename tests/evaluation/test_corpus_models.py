import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    DocumentReviewResult,
    GeneratedDocumentSection,
    GeneratedSourceDocument,
)
from src.rag_platform.evaluation.corpus_validation import (
    load_document_blueprints,
    validate_blueprint_plan,
    validate_generated_document,
)
from src.rag_platform.evaluation.models import SourceDocumentType


BLUEPRINT_PATH = Path("evaluation/blueprints/ecommerce_document_plan.json")


def test_ecommerce_blueprint_contains_exactly_40_controlled_documents() -> None:
    blueprints = load_document_blueprints(BLUEPRINT_PATH)

    report = validate_blueprint_plan(blueprints)

    assert report.is_valid, report.errors
    assert len(blueprints) == 40
    assert {
        doc_type: sum(item.doc_type == doc_type for item in blueprints)
        for doc_type in SourceDocumentType
    } == {
        SourceDocumentType.FAQ: 12,
        SourceDocumentType.SOP: 10,
        SourceDocumentType.RULE: 12,
        SourceDocumentType.MANUAL: 6,
    }


def test_blueprint_plan_has_at_least_four_rule_version_groups() -> None:
    blueprints = load_document_blueprints(BLUEPRINT_PATH)

    version_groups = {
        item.version_group
        for item in blueprints
        if item.doc_type == SourceDocumentType.RULE and item.version_group
    }

    paired_groups = {
        group
        for group in version_groups
        if sum(item.version_group == group for item in blueprints) >= 2
    }
    assert len(paired_groups) >= 4


def test_generated_document_requires_unique_fact_keys() -> None:
    with pytest.raises(ValidationError, match="fact_key"):
        GeneratedSourceDocument(
            source_doc_code="FAQ_ORDER_001",
            title="订单状态与取消常见问题",
            doc_type="FAQ",
            topic="order",
            version="1.0",
            sections=[
                GeneratedDocumentSection(
                    section_code="Q1",
                    heading="订单可以取消吗？",
                    content="待支付订单可以直接取消。",
                    aliases=["怎么取消订单？", "订单不要了怎么办？"],
                    facts=[
                        {
                            "fact_key": "order_cancel",
                            "fact_text": "待支付订单可以直接取消。",
                        }
                    ],
                ),
                GeneratedDocumentSection(
                    section_code="Q2",
                    heading="订单何时不能取消？",
                    content="已完成订单不能取消。",
                    aliases=["完成订单能取消吗？", "订单完成后怎么撤销？"],
                    facts=[
                        {
                            "fact_key": "order_cancel",
                            "fact_text": "已完成订单不能取消。",
                        }
                    ],
                ),
            ],
        )


def test_generated_document_must_cover_blueprint_facts_and_identifiers() -> None:
    blueprint = DocumentBlueprint(
        source_doc_code="RULE_REFUND_001_V2",
        doc_type="RULE",
        title="售后退款规则 V2",
        topic="refund",
        version="2.0",
        effective_from="2026-01-01",
        required_facts=[
            {"fact_key": "refund_window", "description": "退款申请时限"},
            {"fact_key": "coupon_limit", "description": "优惠券订单退款限制"},
        ],
        required_identifiers=["R-REFUND-001", "E-RF-1002"],
        required_sections=["适用范围", "规则条款", "例外条件", "优先级"],
    )
    document = GeneratedSourceDocument(
        source_doc_code=blueprint.source_doc_code,
        title=blueprint.title,
        doc_type=blueprint.doc_type,
        topic=blueprint.topic,
        version=blueprint.version,
        effective_from=blueprint.effective_from,
        sections=[
            GeneratedDocumentSection(
                section_code="1",
                heading="适用范围",
                content="R-REFUND-001 适用于已发货订单。",
                facts=[
                    {
                        "fact_key": "refund_window",
                        "fact_text": "退款申请应在签收后七日内发起。",
                    }
                ],
            ),
            GeneratedDocumentSection(
                section_code="2",
                heading="规则条款",
                content="错误码 E-RF-1002 表示优惠券订单不满足原路退款条件。",
                facts=[],
            ),
        ],
    )

    errors = validate_generated_document(blueprint, document)

    assert any("coupon_limit" in error for error in errors)
    assert not any("R-REFUND-001" in error for error in errors)
    assert not any("E-RF-1002" in error for error in errors)


def test_document_review_pass_condition_is_deterministic() -> None:
    review = DocumentReviewResult(
        source_doc_code="RULE_REFUND_001_V2",
        internal_consistency=0.91,
        fact_coverage=1.0,
        identifier_accuracy=1.0,
        structure_score=0.86,
        version_consistency=0.90,
        ambiguity_risk=0.08,
        overall_score=0.88,
        issues=[],
        summary="审核通过",
    )

    assert review.passed is True

    rejected = review.model_copy(update={"internal_consistency": 0.89})
    assert rejected.passed is False


def test_blueprint_json_is_human_readable_utf8() -> None:
    payload = json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))

    assert payload[0]["title"]
    assert "\\u" not in BLUEPRINT_PATH.read_text(encoding="utf-8")
