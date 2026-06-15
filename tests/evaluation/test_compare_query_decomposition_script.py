from types import SimpleNamespace

from scripts.compare_query_decomposition import (
    build_benchmark_settings,
    build_workflow_snapshot,
    pair_execution_order,
)
from src.rag_platform.core.config import Settings
from src.rag_platform.evaluation.metric_calculator import GoldAnnotations


def test_benchmark_settings_only_toggle_query_decomposition() -> None:
    base = Settings(
        _env_file=None,
        query_analysis_use_llm=True,
        adaptive_retrieval_enabled=True,
        query_decomposition_enabled=True,
    )

    control = build_benchmark_settings(
        base,
        decomposition_enabled=False,
    )
    experiment = build_benchmark_settings(
        base,
        decomposition_enabled=True,
    )

    assert control.query_analysis_use_llm is False
    assert control.adaptive_retrieval_enabled is True
    assert control.query_decomposition_enabled is False
    assert experiment.query_decomposition_enabled is True
    assert base.query_analysis_use_llm is True


def test_workflow_snapshot_records_decomposition_and_coverage() -> None:
    workflow = SimpleNamespace(
        citations=[{"chunk_id": 11}, {"chunk_id": 12}],
        retrieval_round=1,
        query_analysis={"rewritten_query": "固定查询"},
        retrieval_quality={"quality": "GOOD"},
        retrieval_attempts=[],
        decomposition={
            "requires_decomposition": True,
            "sub_queries": [
                {"sub_query_id": "SQ1"},
                {"sub_query_id": "SQ2"},
            ],
        },
        sub_query_coverage={
            "total_sub_queries": 2,
            "covered_sub_queries": 2,
            "coverage_rate": 1.0,
        },
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

    assert snapshot["metrics"]["fact_coverage"] == 1.0
    assert snapshot["decomposition"]["requires_decomposition"] is True
    assert snapshot["sub_query_coverage"]["coverage_rate"] == 1.0


def test_pair_execution_order_is_stable() -> None:
    assert pair_execution_order(
        "CASE_MULTI_HOP_001"
    ) == pair_execution_order("CASE_MULTI_HOP_001")
