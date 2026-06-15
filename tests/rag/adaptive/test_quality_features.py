import pytest

from src.rag_platform.rag.adaptive.quality_features import (
    extract_retrieval_quality_features,
)


def _document(
    *,
    chunk_id: int,
    doc_id: int,
    content: str,
    chunk_type: str = "RULE",
    version: str | None = None,
    sources: list[str] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "page_content": content,
        "chunk_type": chunk_type,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_type": chunk_type,
            "version": version,
            "sources": sources or ["vector"],
        },
    }


def _reranked(
    chunk_id: int,
    score: float,
    content: str = "精排内容",
    version: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "page_content": content,
        "rerank_score": score,
        "metadata": {
            "rerank_score": score,
            "version": version,
        },
    }


def test_extracts_candidate_channel_rerank_and_target_type_features() -> None:
    documents = [
        _document(
            chunk_id=1,
            doc_id=10,
            content="退款规则 R-REFUND-001 适用于普通商品。",
            version="V1",
            sources=["bm25", "vector"],
        ),
        _document(
            chunk_id=2,
            doc_id=11,
            content="新版退款规则调整了处理时限。",
            version="V2",
            sources=["vector"],
        ),
        _document(
            chunk_id=3,
            doc_id=11,
            content="新版规则补充了例外条件。",
            version="V2",
            sources=["bm25"],
        ),
    ]

    features = extract_retrieval_quality_features(
        question="规则 R-REFUND-001 的新旧版本有什么不同？",
        documents=documents,
        reranked_documents=[
            _reranked(
                1,
                0.90,
                "退款规则 R-REFUND-001 适用于普通商品。",
                version="V1",
            ),
            _reranked(2, 0.75, "新版退款规则V2", version="V2"),
            _reranked(3, 0.60, "新版规则V2", version="V2"),
        ],
        target_doc_types=["RULE"],
    )

    assert features.candidate_count == 3
    assert features.distinct_document_count == 2
    assert features.channel_overlap_at_10 == pytest.approx(1 / 3, abs=1e-6)
    assert features.rerank_top1 == 0.90
    assert features.rerank_top3_mean == 0.75
    assert features.rerank_margin == 0.15
    assert features.target_type_coverage == 1.0
    assert features.exact_terms == ["R-REFUND-001"]
    assert features.exact_term_coverage == 1.0
    assert features.distinct_version_count == 2
    assert features.comparison_intent is True


def test_detects_missing_exact_identifier_even_with_high_rerank_score() -> None:
    features = extract_retrieval_quality_features(
        question="错误码 F-ORDER-001 应该怎么处理？",
        documents=[
            _document(
                chunk_id=1,
                doc_id=10,
                content="订单提交失败时请检查库存。",
            )
        ],
        reranked_documents=[_reranked(1, 0.92)],
        target_doc_types=["RULE"],
    )

    assert features.rerank_top1 == 0.92
    assert features.exact_terms == ["F-ORDER-001"]
    assert features.exact_term_coverage == 0.0


def test_extracts_identifier_when_it_is_adjacent_to_chinese_text() -> None:
    features = extract_retrieval_quality_features(
        question="收到错误码F-ORDER-001时应该怎么处理？",
        documents=[
            _document(
                chunk_id=1,
                doc_id=10,
                content="长尾候选包含F-ORDER-001。",
            )
        ],
        reranked_documents=[
            _reranked(1, 0.90, "精排结果不包含目标编号。")
        ],
        target_doc_types=["RULE"],
    )

    assert features.exact_terms == ["F-ORDER-001"]
    assert features.exact_term_coverage == 0.0


def test_detects_single_sided_version_evidence_for_comparison_question() -> None:
    features = extract_retrieval_quality_features(
        question="订单取消规则的新旧版本有什么区别？",
        documents=[
            _document(
                chunk_id=1,
                doc_id=10,
                content="新版订单取消规则。",
                version="V2",
            ),
            _document(
                chunk_id=2,
                doc_id=10,
                content="新版规则的例外条件。",
                version="V2",
            ),
        ],
        reranked_documents=[
            _reranked(1, 0.88, "新版订单规则V2", version="V2"),
            _reranked(2, 0.80, "新版例外条件V2", version="V2"),
        ],
        target_doc_types=["RULE"],
    )

    assert features.comparison_intent is True
    assert features.distinct_version_count == 1


def test_version_coverage_uses_reranked_evidence_not_long_tail() -> None:
    features = extract_retrieval_quality_features(
        question="订单取消规则从V1更新到V2后有什么变化？",
        documents=[
            _document(
                chunk_id=1,
                doc_id=10,
                content="订单取消规则V1。",
                version="1.0",
            ),
            _document(
                chunk_id=2,
                doc_id=11,
                content="订单取消规则V2。",
                version="2.0",
            ),
        ],
        reranked_documents=[
            _reranked(
                2,
                0.92,
                "文档标题：订单取消规则V2。\n本规则替代V1。",
                version="2.0",
            )
        ],
        target_doc_types=["RULE"],
    )

    assert features.comparison_intent is True
    assert features.distinct_version_count == 1
