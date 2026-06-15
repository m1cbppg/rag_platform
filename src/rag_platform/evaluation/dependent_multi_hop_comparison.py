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


def build_dependent_multi_hop_comparison(
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
    case_count = len(cases)
    dependent_sides = [item["dependent"] for item in cases]
    summary = {
        "case_count": case_count,
        "dependent_trigger_count": _count_true(
            dependent_sides,
            "dependent_triggered",
        ),
        "dependent_trigger_rate": _rate(
            dependent_sides,
            "dependent_triggered",
        ),
        "first_hop_gold_hit_rate": _rate(
            dependent_sides,
            "first_hop_gold_hit",
        ),
        "extraction_success_rate": _rate(
            dependent_sides,
            "extraction_success",
        ),
        "supporting_chunk_accuracy_rate": _rate(
            dependent_sides,
            "supporting_chunk_accurate",
        ),
        "intermediate_fact_accuracy_rate": _rate(
            dependent_sides,
            "intermediate_fact_accurate",
        ),
        "second_query_fact_injection_rate": _rate(
            dependent_sides,
            "second_query_contains_fact",
        ),
        "second_hop_gold_hit_rate": _rate(
            dependent_sides,
            "second_hop_gold_hit",
        ),
        "fallback_rate": _rate(
            dependent_sides,
            "fallback_used",
        ),
        "end_to_end_chain_success_rate": _rate(
            dependent_sides,
            "end_to_end_chain_success",
        ),
        "upstream_query_plan_match_rate": _ratio(
            sum(
                item.get("upstream_query_plan_match") is True
                for item in cases
            ),
            case_count,
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
        "latency_ms": _latency_summary(cases),
        "by_chain_type": _by_chain_type(cases),
    }
    return {
        "metadata": metadata,
        "summary": summary,
        "cases": cases,
    }


def render_dependent_multi_hop_comparison_markdown(
    report: dict[str, Any],
) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    lines = [
        "# M10.2 顺序依赖多跳检索成对评测报告",
        "",
        "## 实验范围",
        "",
        f"- 专项数据集：`{metadata.get('dataset_path', '')}`",
        f"- 数据集 SHA256：`{metadata.get('dataset_sha256', '')}`",
        f"- Case 数：{summary['case_count']}",
        "- 两组固定使用规则 Query Analysis、M9 自适应检索和相同召回参数",
        "- 控制组禁止 DEPENDENT 两跳，实验组开启有证据约束的固定两跳",
        "",
        "## 检索指标",
        "",
        "| 指标 | 控制组 | 顺序多跳组 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for metric in RETRIEVAL_METRICS:
        values = summary["metrics"][metric]
        lines.append(
            "| {label} | {control:.4f} | "
            "{dependent:.4f} | {delta:+.4f} |".format(
                label=_METRIC_LABELS[metric],
                **values,
            )
        )

    latency = summary["latency_ms"]
    lines.extend(
        [
            "",
            "## 两跳链路质量",
            "",
            "| 检查项 | 结果 |",
            "|---|---:|",
            f"| DEPENDENT 触发率 | {summary['dependent_trigger_rate']:.2%} |",
            f"| 第一跳 Gold 命中率 | {summary['first_hop_gold_hit_rate']:.2%} |",
            f"| 中间事实抽取成功率 | {summary['extraction_success_rate']:.2%} |",
            f"| 支持 Chunk 准确率 | {summary['supporting_chunk_accuracy_rate']:.2%} |",
            f"| 中间事实准确率 | {summary['intermediate_fact_accuracy_rate']:.2%} |",
            f"| 第二跳事实注入率 | {summary['second_query_fact_injection_rate']:.2%} |",
            f"| 第二跳 Gold 命中率 | {summary['second_hop_gold_hit_rate']:.2%} |",
            f"| 回退率 | {summary['fallback_rate']:.2%} |",
            f"| 端到端链路成功率 | {summary['end_to_end_chain_success_rate']:.2%} |",
            "",
            "## 成本",
            "",
            "- 上游 Query 计划一致率："
            f"{summary['upstream_query_plan_match_rate']:.2%}",
            "- 平均延迟："
            f"{latency['control_mean']:.1f} ms → "
            f"{latency['dependent_mean']:.1f} ms "
            f"（{latency['mean_delta']:+.1f} ms）",
            "- P95 延迟："
            f"{latency['control_p95']:.1f} ms → "
            f"{latency['dependent_p95']:.1f} ms "
            f"（{latency['p95_delta']:+.1f} ms）",
            "",
            "## Case 明细",
            "",
            "| Case | 链路 | 触发 | 中间事实 | 第二跳命中 | "
            "Fact Coverage 差值 | 延迟差值(ms) |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for item in report["cases"]:
        dependent = item["dependent"]
        fact = str(
            dependent.get("dependent_hop", {}).get(
                "intermediate_fact",
                "",
            )
        ).replace("|", "\\|")
        lines.append(
            f"| {item['case_code']} | {item['chain_type']} | "
            f"{_yes_no(dependent.get('dependent_triggered'))} | "
            f"{fact or '-'} | "
            f"{_yes_no(dependent.get('second_hop_gold_hit'))} | "
            f"{item['metric_deltas']['fact_coverage']:+.4f} | "
            f"{item['latency_delta_ms']:+d} |"
        )

    fact = summary["fact_coverage_outcomes"]
    lines.extend(
        [
            "",
            "## 结论数据",
            "",
            "- 事实覆盖率改善 "
            f"{fact['improved']} 条，退化 {fact['regressed']} 条，"
            f"不变 {fact['unchanged']} 条。",
            "- 评测只证明当前专项集上的链路表现；是否发布仍需结合全量回归。",
        ]
    )
    return "\n".join(lines) + "\n"


def _with_deltas(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        **pair,
        "metric_deltas": {
            metric: _round(
                _metric_value(pair["dependent"], metric)
                - _metric_value(pair["control"], metric)
            )
            for metric in RETRIEVAL_METRICS
        },
        "latency_delta_ms": (
            int(pair["dependent"]["latency_ms"])
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
    dependent = _mean(
        [_metric_value(item["dependent"], metric) for item in cases]
    )
    return {
        "control": _round(control),
        "dependent": _round(dependent),
        "delta": _round(dependent - control),
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


def _by_chain_type(
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for chain_type in sorted(
        {str(item["chain_type"]) for item in cases}
    ):
        selected = [
            item
            for item in cases
            if item["chain_type"] == chain_type
        ]
        result[chain_type] = {
            "case_count": len(selected),
            "fact_coverage": _metric_summary(
                selected,
                "fact_coverage",
            ),
            "chain_success_rate": _rate(
                [item["dependent"] for item in selected],
                "end_to_end_chain_success",
            ),
        }
    return result


def _latency_summary(
    cases: list[dict[str, Any]],
) -> dict[str, float]:
    control = [
        float(item["control"]["latency_ms"]) for item in cases
    ]
    dependent = [
        float(item["dependent"]["latency_ms"]) for item in cases
    ]
    control_mean = _mean(control)
    dependent_mean = _mean(dependent)
    control_p95 = _percentile(control, 0.95)
    dependent_p95 = _percentile(dependent, 0.95)
    return {
        "control_mean": _round(control_mean),
        "dependent_mean": _round(dependent_mean),
        "mean_delta": _round(dependent_mean - control_mean),
        "control_p95": _round(control_p95),
        "dependent_p95": _round(dependent_p95),
        "p95_delta": _round(dependent_p95 - control_p95),
    }


def _count_true(
    sides: list[dict[str, Any]],
    key: str,
) -> int:
    return sum(item.get(key) is True for item in sides)


def _rate(
    sides: list[dict[str, Any]],
    key: str,
) -> float:
    return _ratio(_count_true(sides, key), len(sides))


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


def _yes_no(value: Any) -> str:
    return "是" if value is True else "否"
