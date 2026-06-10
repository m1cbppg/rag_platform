from dataclasses import asdict
from typing import Any

from src.rag_platform.evaluation.action_metrics import evaluate_actions


_METRIC_FIELDS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "reciprocal_rank",
    "ndcg_at_5",
    "ndcg_at_10",
    "fact_coverage",
    "citation_precision",
    "citation_recall",
    "faithfulness_score",
    "answer_relevance_score",
    "completeness_score",
    "citation_entailment_score",
    "conflict_handling_score",
)


def build_run_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    action_rows = [
        row
        for row in rows
        if row.get("expected_action") and row.get("actual_action")
    ]
    action_metrics = evaluate_actions(
        expected_actions=[
            row["expected_action"] for row in action_rows
        ],
        actual_actions=[
            row["actual_action"] for row in action_rows
        ],
    )
    latencies = sorted(
        float(row["latency_ms"])
        for row in rows
        if row.get("latency_ms") is not None
    )
    judged = [
        bool(row["judge_passed"])
        for row in rows
        if row.get("judge_passed") is not None
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("case_type") or "UNKNOWN"), []).append(
            row
        )

    return {
        "counts": {
            "total": len(rows),
            "completed": sum(
                row.get("status") == "SUCCESS" for row in rows
            ),
            "failed": sum(
                row.get("status") == "FAILED" for row in rows
            ),
            "errors": sum(
                row.get("actual_action") == "ERROR" for row in rows
            ),
        },
        "metrics": _metric_means(rows),
        "actions": {
            "accuracy": action_metrics.accuracy,
            "confusion_matrix": action_metrics.confusion_matrix,
            "per_action": {
                key: asdict(value)
                for key, value in action_metrics.per_action.items()
            },
        },
        "judge": {
            "evaluated": len(judged),
            "passed": sum(judged),
            "pass_rate": (
                sum(judged) / len(judged) if judged else None
            ),
        },
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "by_case_type": {
            case_type: {
                "count": len(group_rows),
                "metrics": _metric_means(group_rows),
            }
            for case_type, group_rows in sorted(grouped.items())
        },
    }


def _metric_means(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    result = {}
    for field in _METRIC_FIELDS:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None
        ]
        result[field] = sum(values) / len(values) if values else None
    return result


def _percentile(
    values: list[float],
    quantile: float,
) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
