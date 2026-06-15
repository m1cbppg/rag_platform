from pathlib import Path
from types import SimpleNamespace

from scripts.compare_dependent_multi_hop import (
    DependentEvaluationCase,
    build_benchmark_settings,
    build_workflow_snapshot,
    load_dependent_cases,
    pair_execution_order,
)
from src.rag_platform.core.config import Settings


def _case() -> DependentEvaluationCase:
    return DependentEvaluationCase.model_validate(
        {
            "case_code": "DEP_HOP_TEST",
            "chain_type": "测试链路",
            "question": (
                "请先判断是否进入疑似丢失核查，"
                "然后说明如何提交承运商工单？"
            ),
            "expected_intermediate_fact_aliases": [
                "疑似丢失核查"
            ],
            "gold_evidences": [
                {
                    "hop": 1,
                    "chunk_id": 11,
                    "fact_key": "loss_status",
                },
                {
                    "hop": 2,
                    "chunk_id": 12,
                    "fact_key": "loss_ticket",
                },
            ],
        }
    )


def test_benchmark_settings_only_toggle_dependent_execution() -> None:
    base = Settings(
        _env_file=None,
        query_analysis_use_llm=True,
        adaptive_retrieval_enabled=True,
        query_decomposition_enabled=True,
        query_decomposition_allow_dependent=True,
        dependent_multi_hop_enabled=True,
    )

    control = build_benchmark_settings(
        base,
        dependent_enabled=False,
    )
    dependent = build_benchmark_settings(
        base,
        dependent_enabled=True,
    )

    assert control.query_analysis_use_llm is False
    assert control.query_decomposition_enabled is True
    assert control.query_decomposition_allow_dependent is False
    assert control.dependent_multi_hop_enabled is False
    assert dependent.query_decomposition_allow_dependent is True
    assert dependent.dependent_multi_hop_enabled is True


def test_workflow_snapshot_records_complete_dependent_chain() -> None:
    workflow = SimpleNamespace(
        citations=[{"chunk_id": 11}, {"chunk_id": 12}],
        retrieval_round=2,
        query_analysis={"rewritten_query": "固定查询"},
        retrieval_quality={"quality": "GOOD"},
        retrieval_attempts=[
            {
                "round_no": 1,
                "query_variant": "ORIGINAL",
                "documents": [{"chunk_id": 11}],
            },
            {
                "round_no": 2,
                "query_variant": "DEPENDENT_HOP",
                "documents": [{"chunk_id": 12}],
            },
        ],
        decomposition={
            "requires_decomposition": True,
            "decomposition_type": "DEPENDENT",
        },
        dependent_hop={
            "status": "COMPLETED",
            "intermediate_fact": "疑似丢失核查",
            "evidence_quote": "进入疑似丢失核查",
            "supporting_chunk_id": 11,
            "second_hop_query": (
                "疑似丢失核查如何提交承运商工单"
            ),
            "fallback_used": False,
        },
    )

    snapshot = build_workflow_snapshot(
        workflow=workflow,
        case=_case(),
        latency_ms=123,
    )

    assert snapshot["metrics"]["fact_coverage"] == 1.0
    assert snapshot["dependent_triggered"] is True
    assert snapshot["intermediate_fact_accurate"] is True
    assert snapshot["second_hop_gold_hit"] is True
    assert snapshot["end_to_end_chain_success"] is True


def test_specialized_dataset_contains_fifteen_two_hop_cases() -> None:
    path = Path(
        "evaluation/datasets/"
        "rag_dependent_multi_hop_v1.jsonl"
    )

    cases = load_dependent_cases(path)

    assert len(cases) == 15
    assert all(case.first_hop_gold_chunk_ids for case in cases)
    assert all(case.second_hop_gold_chunk_ids for case in cases)


def test_pair_execution_order_is_stable() -> None:
    assert pair_execution_order(
        "DEP_HOP_001"
    ) == pair_execution_order("DEP_HOP_001")
