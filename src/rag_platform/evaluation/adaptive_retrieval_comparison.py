from collections import Counter
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


def build_adaptive_retrieval_comparison(
    *,
    pairs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized_cases = [
        _with_metric_deltas(pair)
        for pair in sorted(
            pairs,
            key=lambda item: str(item["case_code"]),
        )
    ]
    triggered = [
        item
        for item in normalized_cases
        if int(item["adaptive"]["retrieval_rounds"]) > 1
    ]
    strategies = Counter(
        strategy
        for item in triggered
        for strategy in item["adaptive"].get(
            "retry_strategies",
            [],
        )
    )
    case_count = len(normalized_cases)
    summary = {
        "case_count": case_count,
        "triggered_case_count": len(triggered),
        "trigger_rate": _ratio(len(triggered), case_count),
        "strategy_distribution": dict(sorted(strategies.items())),
        "initial_query_plan_match_count": sum(
            1
            for item in normalized_cases
            if item.get("initial_query_plan_match") is True
        ),
        "initial_query_plan_match_rate": _ratio(
            sum(
                1
                for item in normalized_cases
                if item.get("initial_query_plan_match") is True
            ),
            case_count,
        ),
        "metrics": {
            metric: _metric_summary(normalized_cases, metric)
            for metric in RETRIEVAL_METRICS
        },
        "fact_coverage_outcomes": _outcomes(
            normalized_cases,
            "fact_coverage",
        ),
        "recall_at_5_outcomes": _outcomes(
            normalized_cases,
            "recall_at_5",
        ),
        "fully_covered_cases": _fully_covered_summary(
            normalized_cases
        ),
        "triggered_fact_coverage_outcomes": _outcomes(
            triggered,
            "fact_coverage",
        ),
        "strategy_outcomes": _strategy_outcomes(triggered),
        "by_case_type": _by_case_type(normalized_cases),
        "latency_ms": _latency_summary(normalized_cases),
    }
    return {
        "metadata": metadata,
        "summary": summary,
        "cases": normalized_cases,
    }


def render_adaptive_retrieval_comparison_markdown(
    report: dict[str, Any],
) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    lines = [
        "# M9 自适应检索成对评测报告",
        "",
        "## 实验范围",
        "",
        f"- 数据集：`{metadata.get('dataset', '')}`",
        f"- 数据划分：`{metadata.get('split', '')}`",
        f"- Case 数：{summary['case_count']}",
        "- Query 分析：规则模式，控制组与实验组初始计划一致",
        "- 成对执行顺序：按 Case 编码稳定交替，降低冷启动顺序偏差",
        "- 评测范围：仅检索、精排与 Context，不包含答案生成和 Judge",
        "",
        "## 核心结果",
        "",
        "| 指标 | 控制组 | 自适应组 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for metric in RETRIEVAL_METRICS:
        values = summary["metrics"][metric]
        lines.append(
            "| {label} | {control:.4f} | {adaptive:.4f} | "
            "{delta:+.4f} |".format(
                label=_METRIC_LABELS[metric],
                **values,
            )
        )
    latency = summary["latency_ms"]
    lines.extend(
        [
            "",
            "## 触发与成本",
            "",
            f"- 二次检索触发率：{summary['trigger_rate']:.2%} "
            f"（{summary['triggered_case_count']}/"
            f"{summary['case_count']}）",
            "- 初始 Query 计划一致率："
            f"{summary['initial_query_plan_match_rate']:.2%}",
            "- 平均延迟："
            f"{latency['control_mean']:.1f} ms → "
            f"{latency['adaptive_mean']:.1f} ms "
            f"（{latency['mean_delta']:+.1f} ms）",
            "- P95 延迟："
            f"{latency['control_p95']:.1f} ms → "
            f"{latency['adaptive_p95']:.1f} ms "
            f"（{latency['p95_delta']:+.1f} ms）",
            "- 完整事实覆盖 Case："
            f"{summary['fully_covered_cases']['control']} → "
            f"{summary['fully_covered_cases']['adaptive']} "
            f"（{summary['fully_covered_cases']['delta']:+d}）",
            "",
            "### 策略分布",
            "",
            "| 策略 | 次数 | Fact 改善 | Fact 退化 | 平均 Fact 差值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for strategy, outcome in summary["strategy_outcomes"].items():
        fact = outcome["fact_coverage_outcomes"]
        lines.append(
            f"| `{strategy}` | {outcome['count']} | "
            f"{fact['improved']} | {fact['regressed']} | "
            f"{outcome['mean_fact_coverage_delta']:+.4f} |"
        )
    if not summary["strategy_distribution"]:
        lines.append("| 无二次检索 | 0 | 0 | 0 | +0.0000 |")

    lines.extend(
        [
            "",
            "## 分类型结果",
            "",
            "| 类型 | Case 数 | 触发数 | Fact Coverage 控制组 | "
            "自适应组 | 差值 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case_type, item in summary["by_case_type"].items():
        fact = item["fact_coverage"]
        lines.append(
            f"| {case_type} | {item['case_count']} | "
            f"{item['triggered_case_count']} | "
            f"{fact['control']:.4f} | {fact['adaptive']:.4f} | "
            f"{fact['delta']:+.4f} |"
        )

    fact = summary["fact_coverage_outcomes"]
    recall = summary["recall_at_5_outcomes"]
    lines.extend(
        [
            "",
            "## Case 变化",
            "",
            "- 事实覆盖率："
            f"改善 {fact['improved']}，退化 {fact['regressed']}，"
            f"不变 {fact['unchanged']}。",
            "- Recall@5："
            f"改善 {recall['improved']}，退化 {recall['regressed']}，"
            f"不变 {recall['unchanged']}。",
            "",
            "| Case | 类型 | 轮次 | 策略 | Fact Coverage 差值 | "
            "Recall@5 差值 | 延迟差值(ms) |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    ranked_cases = sorted(
        report["cases"],
        key=lambda item: (
            -abs(
                item["metric_deltas"].get(
                    "fact_coverage",
                    0.0,
                )
            ),
            item["case_code"],
        ),
    )
    for item in ranked_cases:
        adaptive = item["adaptive"]
        lines.append(
            "| {case_code} | {case_type} | {rounds} | {strategies} | "
            "{fact:+.4f} | {recall:+.4f} | {latency:+d} |".format(
                case_code=item["case_code"],
                case_type=item["case_type"],
                rounds=adaptive["retrieval_rounds"],
                strategies=", ".join(
                    adaptive.get("retry_strategies", [])
                )
                or "NONE",
                fact=item["metric_deltas"].get(
                    "fact_coverage",
                    0.0,
                ),
                recall=item["metric_deltas"].get(
                    "recall_at_5",
                    0.0,
                ),
                latency=int(item["latency_delta_ms"]),
            )
        )
    return "\n".join(lines) + "\n"


def _with_metric_deltas(pair: dict[str, Any]) -> dict[str, Any]:
    control = pair["control"]
    adaptive = pair["adaptive"]
    deltas = {
        metric: _round(
            _metric_value(adaptive, metric)
            - _metric_value(control, metric)
        )
        for metric in RETRIEVAL_METRICS
    }
    return {
        **pair,
        "metric_deltas": deltas,
        "latency_delta_ms": int(adaptive["latency_ms"])
        - int(control["latency_ms"]),
    }


def _metric_summary(
    cases: list[dict[str, Any]],
    metric: str,
) -> dict[str, float]:
    control = _mean(
        [_metric_value(item["control"], metric) for item in cases]
    )
    adaptive = _mean(
        [_metric_value(item["adaptive"], metric) for item in cases]
    )
    return {
        "control": _round(control),
        "adaptive": _round(adaptive),
        "delta": _round(adaptive - control),
    }


def _outcomes(
    cases: list[dict[str, Any]],
    metric: str,
) -> dict[str, int]:
    improved = 0
    regressed = 0
    for item in cases:
        delta = item["metric_deltas"][metric]
        if delta > 1e-9:
            improved += 1
        elif delta < -1e-9:
            regressed += 1
    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(cases) - improved - regressed,
    }


def _latency_summary(
    cases: list[dict[str, Any]],
) -> dict[str, float]:
    control = [
        float(item["control"]["latency_ms"])
        for item in cases
    ]
    adaptive = [
        float(item["adaptive"]["latency_ms"])
        for item in cases
    ]
    control_mean = _mean(control)
    adaptive_mean = _mean(adaptive)
    control_p95 = _percentile(control, 0.95)
    adaptive_p95 = _percentile(adaptive, 0.95)
    return {
        "control_mean": _round(control_mean),
        "adaptive_mean": _round(adaptive_mean),
        "mean_delta": _round(adaptive_mean - control_mean),
        "control_p50": _round(_percentile(control, 0.50)),
        "adaptive_p50": _round(_percentile(adaptive, 0.50)),
        "control_p95": _round(control_p95),
        "adaptive_p95": _round(adaptive_p95),
        "p95_delta": _round(adaptive_p95 - control_p95),
    }


def _fully_covered_summary(
    cases: list[dict[str, Any]],
) -> dict[str, float | int]:
    control = sum(
        _metric_value(item["control"], "fact_coverage")
        >= 1.0 - 1e-9
        for item in cases
    )
    adaptive = sum(
        _metric_value(item["adaptive"], "fact_coverage")
        >= 1.0 - 1e-9
        for item in cases
    )
    return {
        "control": control,
        "adaptive": adaptive,
        "delta": adaptive - control,
        "control_rate": _ratio(control, len(cases)),
        "adaptive_rate": _ratio(adaptive, len(cases)),
    }


def _strategy_outcomes(
    triggered: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    strategies = sorted(
        {
            strategy
            for item in triggered
            for strategy in item["adaptive"].get(
                "retry_strategies",
                [],
            )
        }
    )
    result = {}
    for strategy in strategies:
        cases = [
            item
            for item in triggered
            if strategy
            in item["adaptive"].get("retry_strategies", [])
        ]
        result[strategy] = {
            "count": len(cases),
            "fact_coverage_outcomes": _outcomes(
                cases,
                "fact_coverage",
            ),
            "mean_fact_coverage_delta": _round(
                _mean(
                    [
                        item["metric_deltas"]["fact_coverage"]
                        for item in cases
                    ]
                )
            ),
            "mean_recall_at_5_delta": _round(
                _mean(
                    [
                        item["metric_deltas"]["recall_at_5"]
                        for item in cases
                    ]
                )
            ),
        }
    return result


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
        triggered = [
            item
            for item in selected
            if int(item["adaptive"]["retrieval_rounds"]) > 1
        ]
        result[case_type] = {
            "case_count": len(selected),
            "triggered_case_count": len(triggered),
            "trigger_rate": _ratio(
                len(triggered),
                len(selected),
            ),
            "fact_coverage": _metric_summary(
                selected,
                "fact_coverage",
            ),
            "recall_at_5": _metric_summary(
                selected,
                "recall_at_5",
            ),
            "fact_coverage_outcomes": _outcomes(
                selected,
                "fact_coverage",
            ),
        }
    return result


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
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float:
    return _round(numerator / denominator) if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)
