from src.rag_platform.evaluation.dependent_multi_hop_comparison import (
    build_dependent_multi_hop_comparison,
    render_dependent_multi_hop_comparison_markdown,
)


def _side(
    fact_coverage: float,
    *,
    latency_ms: int,
    chain_success: bool = False,
) -> dict:
    return {
        "metrics": {
            "recall_at_1": fact_coverage,
            "recall_at_3": fact_coverage,
            "recall_at_5": fact_coverage,
            "recall_at_10": fact_coverage,
            "reciprocal_rank": fact_coverage,
            "ndcg_at_5": fact_coverage,
            "ndcg_at_10": fact_coverage,
            "fact_coverage": fact_coverage,
        },
        "latency_ms": latency_ms,
        "dependent_triggered": chain_success,
        "first_hop_gold_hit": chain_success,
        "extraction_success": chain_success,
        "supporting_chunk_accurate": chain_success,
        "intermediate_fact_accurate": chain_success,
        "second_query_contains_fact": chain_success,
        "second_hop_gold_hit": chain_success,
        "fallback_used": False,
        "end_to_end_chain_success": chain_success,
        "dependent_hop": {
            "intermediate_fact": (
                "疑似丢失核查" if chain_success else ""
            )
        },
    }


def test_builds_dependent_chain_and_retrieval_summary() -> None:
    report = build_dependent_multi_hop_comparison(
        pairs=[
            {
                "case_code": "DEP_HOP_001",
                "chain_type": "物流链路",
                "upstream_query_plan_match": True,
                "control": _side(0.5, latency_ms=100),
                "dependent": _side(
                    1.0,
                    latency_ms=180,
                    chain_success=True,
                ),
            },
            {
                "case_code": "DEP_HOP_002",
                "chain_type": "退款链路",
                "upstream_query_plan_match": True,
                "control": _side(1.0, latency_ms=110),
                "dependent": _side(
                    0.5,
                    latency_ms=190,
                    chain_success=False,
                ),
            },
        ],
        metadata={"dataset_path": "dependent.jsonl"},
    )

    summary = report["summary"]
    assert summary["dependent_trigger_rate"] == 0.5
    assert summary["end_to_end_chain_success_rate"] == 0.5
    assert summary["metrics"]["fact_coverage"] == {
        "control": 0.75,
        "dependent": 0.75,
        "delta": 0.0,
    }
    assert summary["fact_coverage_outcomes"] == {
        "improved": 1,
        "regressed": 1,
        "unchanged": 0,
    }


def test_renders_chinese_dependent_multi_hop_report() -> None:
    report = build_dependent_multi_hop_comparison(
        pairs=[
            {
                "case_code": "DEP_HOP_001",
                "chain_type": "物流链路",
                "upstream_query_plan_match": True,
                "control": _side(0.5, latency_ms=100),
                "dependent": _side(
                    1.0,
                    latency_ms=180,
                    chain_success=True,
                ),
            }
        ],
        metadata={"dataset_path": "dependent.jsonl"},
    )

    markdown = render_dependent_multi_hop_comparison_markdown(
        report
    )

    assert "# M10.2 顺序依赖多跳检索成对评测报告" in markdown
    assert "中间事实抽取成功率" in markdown
    assert "第二跳 Gold 命中率" in markdown
