import json
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    GeneratedSourceDocument,
)
from src.rag_platform.evaluation.corpus_services import (
    CorpusFileStore,
    DocumentGenerationService,
    DocumentReviewService,
)
from src.rag_platform.core.exceptions import ModelResponseFormatError


class FakeJsonClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def _blueprint() -> DocumentBlueprint:
    return DocumentBlueprint(
        source_doc_code="RULE_REFUND_001_V2",
        doc_type="RULE",
        title="售后退款规则 V2",
        topic="refund",
        version="2.0",
        effective_from="2026-01-01",
        required_facts=[
            {"fact_key": "refund_window", "description": "退款申请时限"},
        ],
        required_identifiers=["R-REFUND-001"],
        required_sections=["适用范围", "规则条款", "例外条件", "优先级"],
    )


def _document(
    content: str = "R-REFUND-001 退款申请应在签收后七日内发起。",
) -> dict:
    rule_content = (
        f"{content}\n"
        "本条款适用于平台售后退款业务。\n"
        "规则条款：退款时限按订单签收时间计算。\n"
        "例外条件：质量问题订单可以进入人工复核。\n"
        "优先级：本规则高于普通客服口径。"
    )
    return {
        "source_doc_code": "RULE_REFUND_001_V2",
        "title": "售后退款规则 V2",
        "doc_type": "RULE",
        "topic": "refund",
        "version": "2.0",
        "effective_from": "2026-01-01",
        "effective_to": None,
        "sections": [
            {
                "section_code": "1",
                "heading": "适用范围",
                "content": rule_content,
                "aliases": [],
                "facts": [
                    {
                        "fact_key": "refund_window",
                        "fact_text": "退款申请应在签收后七日内发起。",
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_generation_retries_invalid_model_output_then_returns_document() -> None:
    client = FakeJsonClient(
        [
            {"unexpected": "shape"},
            _document(),
        ]
    )
    service = DocumentGenerationService(
        client=client,
        prompt_template="蓝图：{blueprint_json}\n反馈：{review_feedback}",
        max_attempts=2,
    )

    document = await service.generate(_blueprint())

    assert document.source_doc_code == "RULE_REFUND_001_V2"
    assert len(client.calls) == 2
    assert client.calls[0]["temperature"] == 0.4


@pytest.mark.asyncio
async def test_generation_retries_when_model_returns_invalid_json() -> None:
    client = FakeJsonClient(
        [
            ModelResponseFormatError("模型JSON引号未转义"),
            _document(),
        ]
    )
    service = DocumentGenerationService(
        client=client,
        prompt_template="蓝图：{blueprint_json}\n反馈：{review_feedback}",
        max_attempts=2,
    )

    document = await service.generate(_blueprint())

    assert document.source_doc_code == "RULE_REFUND_001_V2"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_file_store_skips_existing_document_without_force(tmp_path: Path) -> None:
    store = CorpusFileStore(
        source_dir=tmp_path / "source",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    document = GeneratedSourceDocument.model_validate(_document())
    store.save_document(document, generation_round=0)

    assert store.should_generate(document.source_doc_code, force=False) is False
    assert store.should_generate(document.source_doc_code, force=True) is True

    manifest_lines = store.manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    assert json.loads(manifest_lines[0])["status"] == "GENERATED"


@pytest.mark.asyncio
async def test_qwen_review_failure_triggers_deepseek_regeneration() -> None:
    reviewer = FakeJsonClient(
        [
            {
                "source_doc_code": "RULE_REFUND_001_V2",
                "internal_consistency": 0.80,
                "fact_coverage": 1.0,
                "identifier_accuracy": 1.0,
                "structure_score": 0.80,
                "version_consistency": 0.85,
                "ambiguity_risk": 0.20,
                "overall_score": 0.82,
                "issues": ["缺少明确的例外条件"],
                "summary": "需要重写",
            },
            {
                "source_doc_code": "RULE_REFUND_001_V2",
                "internal_consistency": 0.95,
                "fact_coverage": 1.0,
                "identifier_accuracy": 1.0,
                "structure_score": 0.92,
                "version_consistency": 0.94,
                "ambiguity_risk": 0.04,
                "overall_score": 0.93,
                "issues": [],
                "summary": "审核通过",
            },
        ]
    )
    generator_client = FakeJsonClient([_document("R-REFUND-001 退款申请时限明确，例外条件完整。")])
    generator = DocumentGenerationService(
        client=generator_client,
        prompt_template="蓝图：{blueprint_json}\n反馈：{review_feedback}",
    )
    service = DocumentReviewService(
        reviewer=reviewer,
        generator=generator,
        prompt_template=(
            "蓝图：{blueprint_json}\n文档：{document_json}\n"
            "版本上下文：{related_documents_json}"
        ),
        max_regeneration_rounds=2,
    )

    outcome = await service.review(
        blueprint=_blueprint(),
        document=GeneratedSourceDocument.model_validate(_document()),
        related_documents=[],
    )

    assert outcome.review.passed is True
    assert len(outcome.history) == 2
    assert len(generator_client.calls) == 1
    assert "缺少明确的例外条件" in generator_client.calls[0]["user_prompt"]
    assert reviewer.calls[0]["temperature"] == 0


@pytest.mark.asyncio
async def test_qwen_review_retries_invalid_json_without_regenerating() -> None:
    passed_review = {
        "source_doc_code": "RULE_REFUND_001_V2",
        "internal_consistency": 0.95,
        "fact_coverage": 1.0,
        "identifier_accuracy": 1.0,
        "structure_score": 0.92,
        "version_consistency": 0.94,
        "ambiguity_risk": 0.04,
        "overall_score": 0.93,
        "issues": [],
        "summary": "审核通过",
    }
    reviewer = FakeJsonClient(
        [
            ModelResponseFormatError("Qwen JSON格式错误"),
            passed_review,
        ]
    )
    generator_client = FakeJsonClient([])
    service = DocumentReviewService(
        reviewer=reviewer,
        generator=DocumentGenerationService(
            client=generator_client,
            prompt_template="{blueprint_json}\n{review_feedback}",
        ),
        prompt_template="{blueprint_json}\n{document_json}\n{related_documents_json}",
        max_regeneration_rounds=2,
        max_review_attempts=2,
    )

    outcome = await service.review(
        blueprint=_blueprint(),
        document=GeneratedSourceDocument.model_validate(_document()),
        related_documents=[],
    )

    assert outcome.review.passed is True
    assert len(reviewer.calls) == 2
    assert generator_client.calls == []


@pytest.mark.asyncio
async def test_review_stops_after_configured_regeneration_rounds() -> None:
    failed_review = {
        "source_doc_code": "RULE_REFUND_001_V2",
        "internal_consistency": 0.70,
        "fact_coverage": 1.0,
        "identifier_accuracy": 1.0,
        "structure_score": 0.70,
        "version_consistency": 0.70,
        "ambiguity_risk": 0.30,
        "overall_score": 0.70,
        "issues": ["结构不合格"],
        "summary": "拒绝",
    }
    reviewer = FakeJsonClient([failed_review, failed_review, failed_review])
    generator_client = FakeJsonClient([_document(), _document()])
    service = DocumentReviewService(
        reviewer=reviewer,
        generator=DocumentGenerationService(
            client=generator_client,
            prompt_template="{blueprint_json}\n{review_feedback}",
        ),
        prompt_template="{blueprint_json}\n{document_json}\n{related_documents_json}",
        max_regeneration_rounds=2,
    )

    outcome = await service.review(
        blueprint=_blueprint(),
        document=GeneratedSourceDocument.model_validate(_document()),
        related_documents=[],
    )

    assert outcome.review.passed is False
    assert len(outcome.history) == 3
    assert len(generator_client.calls) == 2
