from src.rag_platform.evaluation.adaptive_retrieval_comparison import (
    build_adaptive_retrieval_comparison,
    render_adaptive_retrieval_comparison_markdown,
)


def _side(
    *,
    fact_coverage: float,
    recall_at_5: float,
    retrieval_rounds: int,
    latency_ms: int,
    retry_strategies: list[str] | None = None,
) -> dict:
    return {
        "metrics": {
            "recall_at_1": recall_at_5,
            "recall_at_3": recall_at_5,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_5,
            "reciprocal_rank": recall_at_5,
            "ndcg_at_5": recall_at_5,
            "ndcg_at_10": recall_at_5,
            "fact_coverage": fact_coverage,
        },
        "retrieval_rounds": retrieval_rounds,
        "latency_ms": latency_ms,
        "retry_strategies": retry_strategies or [],
    }


def test_builds_paired_metric_deltas_and_strategy_distribution() -> None:
    pairs = [
        {
            "case_code": "CASE_A",
            "case_type": "DIRECT",
            "initial_query_plan_match": True,
            "control": _side(
                fact_coverage=0.0,
                recall_at_5=0.0,
                retrieval_rounds=1,
                latency_ms=100,
            ),
            "adaptive": _side(
                fact_coverage=1.0,
                recall_at_5=1.0,
                retrieval_rounds=2,
                latency_ms=180,
                retry_strategies=["QUERY_REWRITE"],
            ),
        },
        {
            "case_code": "CASE_B",
            "case_type": "EXACT",
            "initial_query_plan_match": True,
            "control": _side(
                fact_coverage=0.5,
                recall_at_5=0.5,
                retrieval_rounds=1,
                latency_ms=120,
            ),
            "adaptive": _side(
                fact_coverage=1.0,
                recall_at_5=1.0,
                retrieval_rounds=2,
                latency_ms=200,
                retry_strategies=["FORCE_BM25"],
            ),
        },
        {
            "case_code": "CASE_C",
            "case_type": "PARAPHRASE",
            "initial_query_plan_match": True,
            "control": _side(
                fact_coverage=1.0,
                recall_at_5=1.0,
                retrieval_rounds=1,
                latency_ms=90,
            ),
            "adaptive": _side(
                fact_coverage=0.5,
                recall_at_5=0.5,
                retrieval_rounds=2,
                latency_ms=170,
                retry_strategies=["RELAX_FILTER"],
            ),
        },
    ]

    report = build_adaptive_retrieval_comparison(
        pairs=pairs,
        metadata={"dataset": "rag_eval_ecommerce:v2"},
    )

    summary = report["summary"]
    assert summary["case_count"] == 3
    assert summary["triggered_case_count"] == 3
    assert summary["trigger_rate"] == 1.0
    assert summary["strategy_distribution"] == {
        "FORCE_BM25": 1,
        "QUERY_REWRITE": 1,
        "RELAX_FILTER": 1,
    }
    assert summary["fact_coverage_outcomes"] == {
        "improved": 2,
        "regressed": 1,
        "unchanged": 0,
    }
    assert summary["fully_covered_cases"] == {
        "control": 1,
        "adaptive": 2,
        "delta": 1,
        "control_rate": 0.333333,
        "adaptive_rate": 0.666667,
    }
    assert summary["by_case_type"]["DIRECT"][
        "fact_coverage"
    ]["delta"] == 1.0
    assert summary["strategy_outcomes"]["QUERY_REWRITE"][
        "fact_coverage_outcomes"
    ]["improved"] == 1
    assert summary["metrics"]["fact_coverage"] == {
        "control": 0.5,
        "adaptive": 0.833333,
        "delta": 0.333333,
    }
    assert summary["initial_query_plan_match_rate"] == 1.0
    assert report["cases"][0]["metric_deltas"][
        "fact_coverage"
    ] == 1.0


def test_renders_chinese_markdown_report() -> None:
    report = build_adaptive_retrieval_comparison(
        pairs=[
            {
                "case_code": "CASE_A",
                "case_type": "DIRECT",
                "initial_query_plan_match": True,
                "control": _side(
                    fact_coverage=0.0,
                    recall_at_5=0.0,
                    retrieval_rounds=1,
                    latency_ms=100,
                ),
                "adaptive": _side(
                    fact_coverage=1.0,
                    recall_at_5=1.0,
                    retrieval_rounds=2,
                    latency_ms=180,
                    retry_strategies=["QUERY_REWRITE"],
                ),
            }
        ],
        metadata={"dataset": "rag_eval_ecommerce:v2"},
    )

    markdown = render_adaptive_retrieval_comparison_markdown(report)

    assert "# M9 自适应检索成对评测报告" in markdown
    assert "事实覆盖率" in markdown
    assert "QUERY_REWRITE" in markdown
