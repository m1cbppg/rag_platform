from src.rag_platform.rag.adaptive.sub_query_fusion import SubQueryFusion


def _document(
    chunk_id: int,
    score: float,
    *,
    content: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "page_content": content or f"chunk-{chunk_id}",
        "metadata": {"chunk_id": chunk_id},
    }


def test_fusion_preserves_at_least_one_candidate_for_each_sub_query() -> None:
    fusion = SubQueryFusion(rank_constant=60, min_candidates=1)

    result = fusion.fuse(
        [
            {
                "sub_query_id": "SQ1",
                "question": "地址修改条件",
                "documents": [
                    _document(1, 0.99),
                    _document(2, 0.98),
                ],
            },
            {
                "sub_query_id": "SQ2",
                "question": "售后材料要求",
                "documents": [_document(3, 0.4)],
            },
        ],
        top_k=2,
    )

    assert {item["chunk_id"] for item in result} == {1, 3}
    assert result[0]["metadata"]["sub_query_ids"] == ["SQ1"]
    assert result[1]["metadata"]["sub_query_ids"] == ["SQ2"]


def test_fusion_merges_shared_chunk_sub_query_associations_and_score() -> None:
    fusion = SubQueryFusion(rank_constant=60, min_candidates=1)

    result = fusion.fuse(
        [
            {
                "sub_query_id": "SQ1",
                "question": "退款重试条件",
                "documents": [_document(7, 0.8)],
            },
            {
                "sub_query_id": "SQ2",
                "question": "人工复核条件",
                "documents": [_document(7, 0.7)],
            },
        ],
        top_k=5,
    )

    assert len(result) == 1
    assert result[0]["metadata"]["sub_query_ids"] == ["SQ1", "SQ2"]
    assert result[0]["metadata"]["sub_query_texts"] == [
        "退款重试条件",
        "人工复核条件",
    ]
    assert result[0]["score"] > 2 / 62


def test_restore_rerank_quota_recovers_missing_sub_query_candidate() -> None:
    fusion = SubQueryFusion(rank_constant=60, min_candidates=1)
    candidates = fusion.fuse(
        [
            {
                "sub_query_id": "SQ1",
                "question": "地址修改条件",
                "documents": [
                    _document(1, 0.9),
                    _document(2, 0.8),
                ],
            },
            {
                "sub_query_id": "SQ2",
                "question": "售后材料要求",
                "documents": [_document(3, 0.7)],
            },
        ],
        top_k=3,
    )
    reranked = [
        {
            **candidates[0],
            "rerank_score": 0.95,
            "score": 0.95,
        },
        {
            **next(item for item in candidates if item["chunk_id"] == 2),
            "rerank_score": 0.9,
            "score": 0.9,
        },
    ]

    restored = fusion.restore_rerank_quota(
        reranked_documents=reranked,
        candidate_documents=candidates,
        sub_query_ids=["SQ1", "SQ2"],
        top_n=2,
        quota=1,
    )

    assert {item["chunk_id"] for item in restored} == {1, 3}
    restored_sq2 = next(
        item for item in restored if item["chunk_id"] == 3
    )
    assert restored_sq2["metadata"]["quota_restored"] is True


def test_restore_rerank_quota_counts_shared_chunk_for_both_queries() -> None:
    fusion = SubQueryFusion(rank_constant=60, min_candidates=1)
    shared = {
        **_document(9, 0.9),
        "metadata": {
            "chunk_id": 9,
            "sub_query_ids": ["SQ1", "SQ2"],
            "sub_query_texts": ["问题1", "问题2"],
        },
    }

    restored = fusion.restore_rerank_quota(
        reranked_documents=[shared],
        candidate_documents=[shared],
        sub_query_ids=["SQ1", "SQ2"],
        top_n=1,
        quota=1,
    )

    assert [item["chunk_id"] for item in restored] == [9]


def test_restore_rerank_quota_keeps_existing_rerank_order() -> None:
    fusion = SubQueryFusion(rank_constant=60, min_candidates=1)
    reranked = [
        {
            **_document(10, 0.99),
            "metadata": {"anchor_query": True},
        },
        {
            **_document(11, 0.9),
            "metadata": {"sub_query_ids": ["SQ1"]},
        },
        {
            **_document(12, 0.8),
            "metadata": {"sub_query_ids": ["SQ2"]},
        },
    ]

    restored = fusion.restore_rerank_quota(
        reranked_documents=reranked,
        candidate_documents=reranked,
        sub_query_ids=["SQ1", "SQ2"],
        top_n=3,
        quota=1,
    )

    assert [item["chunk_id"] for item in restored] == [10, 11, 12]


def test_calculate_coverage_reports_candidate_and_final_counts() -> None:
    fusion = SubQueryFusion(rank_constant=60, min_candidates=1)
    candidates = [
        {
            **_document(1, 0.9),
            "metadata": {"sub_query_ids": ["SQ1", "SQ2"]},
        },
        {
            **_document(2, 0.8),
            "metadata": {"sub_query_ids": ["SQ1"]},
        },
    ]

    coverage = fusion.calculate_coverage(
        sub_queries=[
            {"sub_query_id": "SQ1", "question": "问题1"},
            {"sub_query_id": "SQ2", "question": "问题2"},
            {"sub_query_id": "SQ3", "question": "问题3"},
        ],
        candidate_documents=candidates,
        final_documents=[candidates[0]],
    )

    assert coverage["total_sub_queries"] == 3
    assert coverage["covered_sub_queries"] == 2
    assert coverage["coverage_rate"] == 2 / 3
    assert coverage["items"]["SQ1"]["candidate_count"] == 2
    assert coverage["items"]["SQ2"]["final_count"] == 1
    assert coverage["items"]["SQ3"]["covered"] is False
