import pytest
from langchain_core.documents import Document

from src.rag_platform.rag.adaptive.multi_round_fusion import (
    MultiRoundFusion,
)
from src.rag_platform.rag.retrievers.document_mapper import (
    RetrievalDocumentMapper,
)
from src.rag_platform.domain.search import RetrievalHit


def _document(chunk_id: int, score: float, source: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "source": source,
        "page_content": f"chunk-{chunk_id}",
        "metadata": {
            "chunk_id": chunk_id,
            "score": score,
            "source": source,
            "doc_id": chunk_id + 100,
        },
    }


def test_weighted_rrf_merges_rounds_and_preserves_provenance() -> None:
    fusion = MultiRoundFusion(rank_constant=60)

    result = fusion.fuse(
        [
            {
                "round_no": 1,
                "strategy": "INITIAL",
                "query_variant": "ORIGINAL",
                "queries": ["订单取消规则"],
                "weight": 1.0,
                "documents": [
                    _document(1, 0.9, "hybrid"),
                    _document(2, 0.8, "hybrid"),
                ],
            },
            {
                "round_no": 2,
                "strategy": "QUERY_REWRITE",
                "query_variant": "REWRITTEN",
                "queries": ["订单取消规则 新版 旧版"],
                "weight": 0.9,
                "documents": [
                    _document(2, 0.95, "hybrid"),
                    _document(3, 0.7, "hybrid"),
                ],
            },
        ],
        top_k=10,
    )

    assert [item["chunk_id"] for item in result] == [2, 1, 3]
    repeated = result[0]
    assert repeated["score"] == pytest.approx(
        1.0 / 62 + 0.9 / 61
    )
    assert repeated["source"] == "adaptive"
    assert repeated["metadata"]["retrieval_rounds"] == [1, 2]
    assert len(repeated["metadata"]["adaptive_sources"]) == 2


def test_document_mapper_keeps_hybrid_ranking_metadata() -> None:
    mapper = RetrievalDocumentMapper()
    document = mapper.to_document(
        RetrievalHit(
            chunk_id=7,
            score=0.03,
            source="hybrid",
            metadata={
                "content": "退款规则",
                "doc_id": 10,
                "sources": ["vector", "bm25"],
                "vector_rank": 2,
                "vector_raw_score": 0.82,
                "bm25_rank": 1,
                "bm25_raw_score": 7.5,
                "rrf_score": 0.03,
            },
        )
    )

    assert isinstance(document, Document)
    assert document.metadata["sources"] == ["vector", "bm25"]
    assert document.metadata["vector_rank"] == 2
    assert document.metadata["bm25_rank"] == 1
    assert document.metadata["rrf_score"] == 0.03
