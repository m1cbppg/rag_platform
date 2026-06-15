import math
from typing import Any


RETRIEVAL_METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "reciprocal_rank",
    "ndcg_at_5",
    "ndcg_at_10",
    "fact_coverage",
)

_METRIC_LABELS = {
    "recall_at_1": "Recall@1",
    "recall_at_3": "Recall@3",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "reciprocal_rank": "MRR",
    "ndcg_at_5": "nDCG@5",
    "ndcg_at_10": "nDCG@10",
    "fact_coverage": "事实覆盖率",
}


def build_query_decomposition_comparison(
    *,
    pairs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    cases = [
        _with_deltas(item)
        for item in sorted(
            pairs,
            key=lambda value: str(value["case_code"]),
        )
    ]
    triggered = [
        item
        for item in cases
        if item["decomposition"]
        .get("decomposition", {})
        .get("requires_decomposition")
    ]
    case_count = len(cases)
    trigger_count = len(triggered)
    summary = {
        "case_count": case_count,
        "triggered_case_count": trigger_count,
        "trigger_rate": _ratio(trigger_count, case_count),
        "upstream_query_plan_match_rate": _ratio(
            sum(
                item.get("upstream_query_plan_match") is True
                for item in cases
            ),
            case_count,
        ),
        "mean_sub_query_count": _round(
            _mean(
                [
                    len(
                        item["decomposition"]
                        .get("decomposition", {})
                        .get("sub_queries", [])
                    )
                    for item in triggered
                ]
            )
        ),
        "mean_sub_query_coverage": _round(
            _mean(
                [
                    float(
                        item["decomposition"]
                        .get("sub_query_coverage", {})
                        .get("coverage_rate", 0.0)
                    )
                    for item in triggered
                ]
            )
        ),
        "fully_covered_sub_query_cases": sum(
            float(
                item["decomposition"]
                .get("sub_query_coverage", {})
                .get("coverage_rate", 0.0)
            )
            >= 1.0 - 1e-9
            for item in triggered
        ),
        "metrics": {
            metric: _metric_summary(cases, metric)
            for metric in RETRIEVAL_METRICS
        },
        "fact_coverage_outcomes": _outcomes(
            cases,
            "fact_coverage",
        ),
        "recall_at_5_outcomes": _outcomes(
            cases,
            "recall_at_5",
        ),
        "by_case_type": _by_case_type(cases),
        "latency_ms": _latency_summary(cases),
    }
    return {
        "metadata": metadata,
        "summary": summary,
        "cases": cases,
    }


def render_query_decomposition_comparison_markdown(
    report: dict[str, Any],
) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    lines = [
        "# M10 查询分解成对评测报告",
        "",
        "## 实验范围",
        "",
        f"- 数据集：`{metadata.get('dataset', '')}`",
        f"- 数据划分：`{metadata.get('split', '')}`",
        f"- Case 数：{summary['case_count']}",
        "- 两组均开启 M9 自适应检索并固定规则 Query Analysis",
        "- 控制组关闭查询分解，实验组开启查询分解",
        "",
        "## 核心结果",
        "",
        "| 指标 | M9 控制组 | M10 分解组 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for metric in RETRIEVAL_METRICS:
        values = summary["metrics"][metric]
        lines.append(
            "| {label} | {control:.4f} | "
            "{decomposition:.4f} | {delta:+.4f} |".format(
                label=_METRIC_LABELS[metric],
                **values,
            )
        )
    latency = summary["latency_ms"]
    lines.extend(
        [
            "",
            "## 分解触发与成本",
            "",
            "- 分解触发率："
            f"{summary['trigger_rate']:.2%} "
            f"（{summary['triggered_case_count']}/"
            f"{summary['case_count']}）",
            "- 上游 Query 计划一致率："
            f"{summary['upstream_query_plan_match_rate']:.2%}",
            "- 平均子问题数："
            f"{summary['mean_sub_query_count']:.2f}",
            "- 平均子问题覆盖率："
            f"{summary['mean_sub_query_coverage']:.2%}",
            "- 子问题完全覆盖 Case："
            f"{summary['fully_covered_sub_query_cases']}/"
            f"{summary['triggered_case_count']}",
            "- 平均延迟："
            f"{latency['control_mean']:.1f} ms → "
            f"{latency['decomposition_mean']:.1f} ms "
            f"（{latency['mean_delta']:+.1f} ms）",
            "- P95 延迟："
            f"{latency['control_p95']:.1f} ms → "
            f"{latency['decomposition_p95']:.1f} ms "
            f"（{latency['p95_delta']:+.1f} ms）",
            "",
            "## 分类型结果",
            "",
            "| 类型 | Case 数 | 触发数 | Fact Coverage 控制组 | "
            "分解组 | 差值 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case_type, item in summary["by_case_type"].items():
        fact = item["fact_coverage"]
        lines.append(
            f"| {case_type} | {item['case_count']} | "
            f"{item['triggered_case_count']} | "
            f"{fact['control']:.4f} | "
            f"{fact['decomposition']:.4f} | "
            f"{fact['delta']:+.4f} |"
        )
    fact = summary["fact_coverage_outcomes"]
    lines.extend(
        [
            "",
            "## Case 变化",
            "",
            "- 事实覆盖率："
            f"改善 {fact['improved']}，退化 {fact['regressed']}，"
            f"不变 {fact['unchanged']}。",
            "",
            "| Case | 类型 | 是否分解 | 子问题覆盖率 | "
            "Fact Coverage 差值 | 延迟差值(ms) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in sorted(
        report["cases"],
        key=lambda value: (
            -abs(value["metric_deltas"]["fact_coverage"]),
            value["case_code"],
        ),
    ):
        decomposition = item["decomposition"]
        triggered = decomposition.get(
            "decomposition",
            {},
        ).get("requires_decomposition", False)
        coverage = decomposition.get(
            "sub_query_coverage",
            {},
        ).get("coverage_rate", 1.0)
        lines.append(
            f"| {item['case_code']} | {item['case_type']} | "
            f"{'是' if triggered else '否'} | "
            f"{float(coverage):.2%} | "
            f"{item['metric_deltas']['fact_coverage']:+.4f} | "
            f"{item['latency_delta_ms']:+d} |"
        )
    return "\n".join(lines) + "\n"


def _with_deltas(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        **pair,
        "metric_deltas": {
            metric: _round(
                _metric_value(pair["decomposition"], metric)
                - _metric_value(pair["control"], metric)
            )
            for metric in RETRIEVAL_METRICS
        },
        "latency_delta_ms": (
            int(pair["decomposition"]["latency_ms"])
            - int(pair["control"]["latency_ms"])
        ),
    }


def _metric_summary(
    cases: list[dict[str, Any]],
    metric: str,
) -> dict[str, float]:
    control = _mean(
        [_metric_value(item["control"], metric) for item in cases]
    )
    decomposition = _mean(
        [
            _metric_value(item["decomposition"], metric)
            for item in cases
        ]
    )
    return {
        "control": _round(control),
        "decomposition": _round(decomposition),
        "delta": _round(decomposition - control),
    }


def _outcomes(
    cases: list[dict[str, Any]],
    metric: str,
) -> dict[str, int]:
    improved = sum(
        item["metric_deltas"][metric] > 1e-9 for item in cases
    )
    regressed = sum(
        item["metric_deltas"][metric] < -1e-9 for item in cases
    )
    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(cases) - improved - regressed,
    }


def _by_case_type(
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for case_type in sorted(
        {str(item["case_type"]) for item in cases}
    ):
        selected = [
            item
            for item in cases
            if item["case_type"] == case_type
        ]
        result[case_type] = {
            "case_count": len(selected),
            "triggered_case_count": sum(
                item["decomposition"]
                .get("decomposition", {})
                .get("requires_decomposition", False)
                for item in selected
            ),
            "fact_coverage": _metric_summary(
                selected,
                "fact_coverage",
            ),
            "recall_at_5": _metric_summary(
                selected,
                "recall_at_5",
            ),
        }
    return result


def _latency_summary(
    cases: list[dict[str, Any]],
) -> dict[str, float]:
    control = [
        float(item["control"]["latency_ms"]) for item in cases
    ]
    decomposition = [
        float(item["decomposition"]["latency_ms"])
        for item in cases
    ]
    control_mean = _mean(control)
    decomposition_mean = _mean(decomposition)
    control_p95 = _percentile(control, 0.95)
    decomposition_p95 = _percentile(decomposition, 0.95)
    return {
        "control_mean": _round(control_mean),
        "decomposition_mean": _round(decomposition_mean),
        "mean_delta": _round(
            decomposition_mean - control_mean
        ),
        "control_p95": _round(control_p95),
        "decomposition_p95": _round(decomposition_p95),
        "p95_delta": _round(
            decomposition_p95 - control_p95
        ),
    }


def _metric_value(side: dict[str, Any], metric: str) -> float:
    value = side.get("metrics", {}).get(metric)
    return float(value) if value is not None else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def _ratio(numerator: int, denominator: int) -> float:
    return _round(numerator / denominator) if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)
