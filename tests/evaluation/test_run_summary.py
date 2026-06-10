import pytest

from src.rag_platform.evaluation.run_summary import build_run_summary


def test_summary_excludes_none_metrics_and_builds_action_matrix() -> None:
    summary = build_run_summary(
        [
            {
                "case_type": "DIRECT",
                "expected_action": "ANSWER",
                "actual_action": "ANSWER",
                "status": "SUCCESS",
                "recall_at_5": 1.0,
                "fact_coverage": 1.0,
                "latency_ms": 100,
                "judge_passed": 1,
            },
            {
                "case_type": "NO_ANSWER",
                "expected_action": "REFUSE",
                "actual_action": "ANSWER",
                "status": "SUCCESS",
                "recall_at_5": None,
                "fact_coverage": None,
                "latency_ms": 300,
                "judge_passed": 0,
            },
            {
                "case_type": "DIRECT",
                "expected_action": "ANSWER",
                "actual_action": "ERROR",
                "status": "FAILED",
                "recall_at_5": 0.0,
                "fact_coverage": 0.0,
                "latency_ms": 200,
                "judge_passed": None,
            },
        ]
    )

    assert summary["counts"] == {
        "total": 3,
        "completed": 2,
        "failed": 1,
        "errors": 1,
    }
    assert summary["metrics"]["recall_at_5"] == 0.5
    assert summary["metrics"]["fact_coverage"] == 0.5
    assert summary["judge"]["pass_rate"] == 0.5
    assert summary["latency_ms"]["p50"] == 200
    assert summary["latency_ms"]["p95"] == pytest.approx(290)
    assert summary["actions"]["confusion_matrix"]["REFUSE"]["ANSWER"] == 1
    assert summary["by_case_type"]["DIRECT"]["count"] == 2


def test_summary_handles_empty_run() -> None:
    summary = build_run_summary([])

    assert summary["counts"]["total"] == 0
    assert summary["metrics"]["recall_at_5"] is None
    assert summary["latency_ms"]["p50"] is None
