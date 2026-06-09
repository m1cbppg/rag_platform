import hashlib
import json
from pathlib import Path

from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    DocumentReviewResult,
    GeneratedSourceDocument,
    ReviewedDocumentOutcome,
    ReviewHistoryItem,
)
from src.rag_platform.evaluation.corpus_persistence import (
    build_source_document_spec,
)
from src.rag_platform.evaluation.models import ReviewStatus


def test_build_source_document_spec_preserves_review_history(
    tmp_path: Path,
) -> None:
    blueprint = DocumentBlueprint(
        source_doc_code="FAQ_TEST_001",
        doc_type="FAQ",
        title="测试FAQ",
        topic="test",
        version="1.0",
        required_facts=[
            {"fact_key": "fact_1", "description": "测试事实"},
        ],
        required_identifiers=["F-TEST-001"],
        required_sections=["测试问题"],
    )
    document = GeneratedSourceDocument.model_validate(
        {
            "source_doc_code": "FAQ_TEST_001",
            "title": "测试FAQ",
            "doc_type": "FAQ",
            "topic": "test",
            "version": "1.0",
            "sections": [
                {
                    "section_code": "Q1",
                    "heading": "测试问题",
                    "content": "F-TEST-001 测试答案。",
                    "aliases": ["测试怎么处理？", "如何处理测试？"],
                    "facts": [
                        {"fact_key": "fact_1", "fact_text": "测试事实"}
                    ],
                }
            ],
        }
    )
    review = DocumentReviewResult(
        source_doc_code="FAQ_TEST_001",
        internal_consistency=0.95,
        fact_coverage=1.0,
        identifier_accuracy=1.0,
        structure_score=0.92,
        version_consistency=0.90,
        ambiguity_risk=0.05,
        overall_score=0.93,
        issues=[],
        summary="通过",
    )
    outcome = ReviewedDocumentOutcome(
        document=document,
        review=review,
        history=[ReviewHistoryItem(round_no=0, review=review)],
    )
    source_path = tmp_path / "FAQ_TEST_001.json"
    source_content = (
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    source_path.write_text(source_content, encoding="utf-8")

    spec = build_source_document_spec(
        blueprint=blueprint,
        document=document,
        source_path=source_path,
        outcome=outcome,
        project_root=tmp_path,
    )

    assert spec.review_status == ReviewStatus.PASSED
    assert spec.review_score == 0.93
    assert spec.generation_spec["review_history"][0]["round_no"] == 0
    assert spec.relative_file_path == "FAQ_TEST_001.json"
    assert spec.source_content_sha256 == hashlib.sha256(
        source_content.encode("utf-8")
    ).hexdigest()
