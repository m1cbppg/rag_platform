from src.rag_platform.evaluation.query_decomposition_comparison import (
    build_query_decomposition_comparison,
    render_query_decomposition_comparison_markdown,
)


def side(
    fact_coverage: float,
    *,
    latency_ms: int,
    triggered: bool = False,
    coverage_rate: float = 1.0,
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
        "decomposition": {
            "requires_decomposition": triggered,
            "sub_queries": (
                [
                    {"sub_query_id": "SQ1"},
                    {"sub_query_id": "SQ2"},
                ]
                if triggered
                else []
            ),
        },
        "sub_query_coverage": {
            "coverage_rate": coverage_rate,
        },
    }


def test_builds_decomposition_trigger_coverage_and_metric_deltas() -> None:
    report = build_query_decomposition_comparison(
        pairs=[
            {
                "case_code": "CASE_A",
                "case_type": "MULTI_HOP",
                "upstream_query_plan_match": True,
                "control": side(0.5, latency_ms=100),
                "decomposition": side(
                    1.0,
                    latency_ms=180,
                    triggered=True,
                    coverage_rate=1.0,
                ),
            },
            {
                "case_code": "CASE_B",
                "case_type": "MULTI_CONDITION",
                "upstream_query_plan_match": True,
                "control": side(1.0, latency_ms=120),
                "decomposition": side(
                    0.5,
                    latency_ms=170,
                    triggered=True,
                    coverage_rate=0.5,
                ),
            },
        ],
        metadata={"dataset": "rag_eval_ecommerce:v2"},
    )

    summary = report["summary"]
    assert summary["trigger_rate"] == 1.0
    assert summary["mean_sub_query_count"] == 2.0
    assert summary["mean_sub_query_coverage"] == 0.75
    assert summary["metrics"]["fact_coverage"] == {
        "control": 0.75,
        "decomposition": 0.75,
        "delta": 0.0,
    }
    assert summary["fact_coverage_outcomes"] == {
        "improved": 1,
        "regressed": 1,
        "unchanged": 0,
    }
    assert summary["by_case_type"]["MULTI_HOP"][
        "fact_coverage"
    ]["delta"] == 0.5


def test_renders_chinese_query_decomposition_report() -> None:
    report = build_query_decomposition_comparison(
        pairs=[
            {
                "case_code": "CASE_A",
                "case_type": "MULTI_HOP",
                "upstream_query_plan_match": True,
                "control": side(0.5, latency_ms=100),
                "decomposition": side(
                    1.0,
                    latency_ms=180,
                    triggered=True,
                ),
            }
        ],
        metadata={"dataset": "rag_eval_ecommerce:v2"},
    )

    markdown = render_query_decomposition_comparison_markdown(
        report
    )

    assert "# M10 查询分解成对评测报告" in markdown
    assert "子问题覆盖率" in markdown
    assert "MULTI_HOP" in markdown
