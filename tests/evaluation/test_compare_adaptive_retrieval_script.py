from types import SimpleNamespace

from scripts.compare_adaptive_retrieval import (
    build_benchmark_settings,
    build_workflow_snapshot,
    pair_execution_order,
)
from src.rag_platform.core.config import Settings
from src.rag_platform.evaluation.metric_calculator import GoldAnnotations


def test_benchmark_settings_fix_query_analysis_and_toggle_adaptive() -> None:
    base = Settings(
        _env_file=None,
        query_analysis_use_llm=True,
        adaptive_retrieval_enabled=True,
    )

    control = build_benchmark_settings(
        base,
        adaptive_enabled=False,
    )
    adaptive = build_benchmark_settings(
        base,
        adaptive_enabled=True,
    )

    assert control.query_analysis_use_llm is False
    assert control.adaptive_retrieval_enabled is False
    assert adaptive.query_analysis_use_llm is False
    assert adaptive.adaptive_retrieval_enabled is True
    assert base.query_analysis_use_llm is True


def test_workflow_snapshot_records_metrics_and_retry_strategy() -> None:
    workflow = SimpleNamespace(
        citations=[
            {"chunk_id": 11},
            {"chunk_id": 12},
            {"chunk_id": 11},
        ],
        retrieval_round=2,
        query_analysis={"rewritten_query": "固定查询"},
        retrieval_quality={"quality": "GOOD"},
        retrieval_attempts=[
            {
                "round_no": 1,
                "strategy": "INITIAL",
                "query_variant": "ORIGINAL",
                "queries": ["固定查询"],
                "retrieval_mode": "hybrid",
                "doc_type_filter": "FAQ",
                "business_domain_filter": "order",
                "removed_filters": [],
                "quality": {
                    "quality": "WEAK",
                    "retry_strategy": "FORCE_BM25",
                },
                "reranked_documents": [
                    {"chunk_id": 11, "rerank_score": 0.7}
                ],
            },
            {
                "round_no": 2,
                "strategy": "FORCE_BM25",
                "query_variant": "EXACT",
                "queries": ["F-001 固定查询"],
                "retrieval_mode": "bm25",
                "doc_type_filter": None,
                "business_domain_filter": "order",
                "removed_filters": ["doc_type"],
                "quality": {"quality": "GOOD"},
                "reranked_documents": [
                    {"chunk_id": 12, "rerank_score": 0.8}
                ],
            },
        ],
    )
    gold = GoldAnnotations(
        relevance_by_chunk={11: 3, 12: 3},
        fact_keys_by_chunk={11: {"fact_a"}, 12: {"fact_b"}},
    )

    snapshot = build_workflow_snapshot(
        workflow=workflow,
        gold=gold,
        latency_ms=123,
    )

    assert snapshot["retrieved_chunk_ids"] == [11, 12]
    assert snapshot["metrics"]["fact_coverage"] == 1.0
    assert snapshot["retrieval_rounds"] == 2
    assert snapshot["retry_strategies"] == ["FORCE_BM25"]
    assert snapshot["attempts"][1]["removed_filters"] == ["doc_type"]


def test_pair_execution_order_is_stable_and_balanced() -> None:
    first = pair_execution_order("CASE_001")

    assert pair_execution_order("CASE_001") == first
    orders = {
        pair_execution_order(f"CASE_{index:03d}")
        for index in range(20)
    }
    assert orders == {
        ("control", "adaptive"),
        ("adaptive", "control"),
    }
