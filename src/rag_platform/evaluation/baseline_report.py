from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.rag_platform.evaluation.failure_attribution import (
    AttributionCode,
    attribute_case,
    attribution_label,
    attribution_recommendation,
)
from src.rag_platform.evaluation.run_summary import build_run_summary


def build_baseline_report(
    *,
    run: dict[str, Any],
    case_results: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    system_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hits_by_result: dict[int, list[dict[str, Any]]] = {}
    for hit in hits:
        hits_by_result.setdefault(
            int(hit["case_result_id"]),
            [],
        ).append(hit)
    evidences_by_case: dict[int, list[dict[str, Any]]] = {}
    for evidence in evidences:
        evidences_by_case.setdefault(
            int(evidence["case_id"]),
            [],
        ).append(evidence)

    case_entries = []
    for result in case_results:
        attribution = attribute_case(
            case_result=result,
            hits=hits_by_result.get(int(result["id"]), []),
            evidences=evidences_by_case.get(int(result["case_id"]), []),
        )
        case_entries.append(
            {
                **_json_value(result),
                "retrieval_hits": _json_value(
                    hits_by_result.get(int(result["id"]), [])
                ),
                "gold_evidences": _json_value(
                    evidences_by_case.get(int(result["case_id"]), [])
                ),
                "attribution": attribution.model_dump(mode="json"),
            }
        )

    total = len(case_entries)
    passed = sum(
        item["attribution"]["primary_code"] == AttributionCode.PASS.value
        for item in case_entries
    )
    attribution_counts = Counter(
        item["attribution"]["primary_code"]
        for item in case_entries
    )
    attribution_summary = {
        code: {
            "label": attribution_label(code),
            "count": count,
            "share": count / total if total else None,
            "recommendation": attribution_recommendation(code),
        }
        for code, count in sorted(
            attribution_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }
    priorities = [
        {
            "rank": rank,
            "code": code,
            **details,
        }
        for rank, (code, details) in enumerate(
            (
                (code, details)
                for code, details in attribution_summary.items()
                if code != AttributionCode.PASS.value
            ),
            start=1,
        )
    ]
    summary = run.get("summary_metrics") or build_run_summary(case_results)
    normalized_diagnostics = _json_value(system_diagnostics or {})
    return {
        "report_version": "m7-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": _json_value(run),
        "overview": {
            "total_cases": total,
            "attribution_passed": passed,
            "attribution_pass_rate": passed / total if total else None,
            "judge_pass_rate": _mean(
                [
                    float(item["judge_passed"])
                    for item in case_entries
                    if item.get("judge_passed") is not None
                ]
            ),
            "action_accuracy": _mean(
                [
                    float(item["action_correct"])
                    for item in case_entries
                    if item.get("action_correct") is not None
                ]
            ),
        },
        "metrics": _json_value(summary),
        "system_diagnostics": normalized_diagnostics,
        "system_findings": _system_findings(normalized_diagnostics),
        "attribution_summary": attribution_summary,
        "stage_funnel": {
            "merged_fact_coverage": _stage_mean(
                case_entries,
                "merged",
            ),
            "rerank_fact_coverage": _stage_mean(
                case_entries,
                "rerank",
            ),
            "final_fact_coverage": _stage_mean(
                case_entries,
                "final",
            ),
        },
        "by_case_type": _slice_summary(case_entries, "case_type"),
        "by_difficulty": _slice_summary(case_entries, "difficulty"),
        "priorities": priorities,
        "cases": case_entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    overview = report["overview"]
    funnel = report["stage_funnel"]
    lines = [
        f"# {run['run_code']} RAG 基线评测报告",
        "",
        "## 运行信息",
        "",
        f"- 实验版本：`{run.get('experiment_version')}`",
        f"- 实验名称：`{run.get('experiment_name')}`",
        f"- 运行状态：`{run.get('status')}`",
        f"- 总题数：{overview['total_cases']}",
        f"- 归因通过率：{_percent(overview['attribution_pass_rate'])}",
        f"- Judge 通过率：{_percent(overview['judge_pass_rate'])}",
        f"- 行为准确率：{_percent(overview['action_accuracy'])}",
        "",
        "## 系统级根因",
        "",
    ]
    if not report["system_findings"]:
        lines.append("没有发现可由当前诊断数据直接确认的系统级配置问题。")
    for finding in report["system_findings"]:
        lines.extend(
            [
                f"### {finding['severity']}：{finding['title']}",
                "",
                finding["evidence"],
                "",
                f"处理建议：{finding['recommendation']}",
                "",
            ]
        )
    lines.extend(
        [
        "## 检索漏斗",
        "",
        "| 阶段 | 必要事实平均覆盖率 |",
        "| --- | ---: |",
        f"| 融合召回 | {_percent(funnel['merged_fact_coverage'])} |",
        f"| 精排结果 | {_percent(funnel['rerank_fact_coverage'])} |",
        f"| 最终 Context | {_percent(funnel['final_fact_coverage'])} |",
        "",
        "## 失败归因分布",
        "",
        "| 归因 | 数量 | 占比 | 优化建议 |",
        "| --- | ---: | ---: | --- |",
        ]
    )
    for details in report["attribution_summary"].values():
        lines.append(
            "| {label} | {count} | {share} | {recommendation} |".format(
                label=_escape(details["label"]),
                count=details["count"],
                share=_percent(details["share"]),
                recommendation=_escape(details["recommendation"]),
            )
        )

    lines.extend(
        [
            "",
            "## 题型表现",
            "",
            "| 题型 | 题数 | 归因通过率 | Judge通过率 | Recall@10 | Fact Coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, details in report["by_case_type"].items():
        lines.append(_slice_row(name, details))

    lines.extend(
        [
            "",
            "## 难度表现",
            "",
            "| 难度 | 题数 | 归因通过率 | Judge通过率 | Recall@10 | Fact Coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, details in report["by_difficulty"].items():
        lines.append(_slice_row(name, details))

    lines.extend(["", "## V0 后续优化优先级", ""])
    if not report["priorities"]:
        lines.append("当前运行没有归因失败项。")
    for item in report["priorities"]:
        lines.append(
            f"{item['rank']}. **{item['label']}**："
            f"{item['count']} 题，占 {_percent(item['share'])}。"
            f"{item['recommendation']}"
        )

    failed_cases = [
        item
        for item in report["cases"]
        if item["attribution"]["primary_code"] != AttributionCode.PASS.value
    ]
    lines.extend(
        [
            "",
            "## 典型失败案例",
            "",
            "| Case | 题型 | 难度 | 问题 | 预期/实际 | 主因 | Recall@10 | Fact Coverage |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for item in failed_cases[:30]:
        attribution = item["attribution"]
        lines.append(
            "| {case_code} | {case_type} | {difficulty} | {question} | "
            "{expected}/{actual} | {label} | {recall} | {coverage} |".format(
                case_code=_escape(item.get("case_code")),
                case_type=_escape(item.get("case_type")),
                difficulty=_escape(item.get("difficulty")),
                question=_escape(_truncate(item.get("question"), 80)),
                expected=_escape(item.get("expected_action")),
                actual=_escape(item.get("actual_action")),
                label=_escape(attribution["primary_label"]),
                recall=_number(item.get("recall_at_10")),
                coverage=_number(item.get("fact_coverage")),
            )
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 运行状态 SUCCESS 表示评测流程完成，不表示答案质量通过。",
            "- 主归因采用上游优先原则，避免把召回问题重复计算为答案问题。",
            "- JSON 报告包含全部逐题证据、阶段覆盖率和次级归因。",
            "",
        ]
    )
    return "\n".join(lines)


def _slice_summary(
    cases: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in cases:
        grouped.setdefault(str(item.get(field) or "UNKNOWN"), []).append(item)
    result = {}
    for name, rows in sorted(grouped.items()):
        result[name] = {
            "count": len(rows),
            "pass_rate": _mean(
                [
                    float(
                        row["attribution"]["primary_code"]
                        == AttributionCode.PASS.value
                    )
                    for row in rows
                ]
            ),
            "judge_pass_rate": _mean(
                [
                    float(row["judge_passed"])
                    for row in rows
                    if row.get("judge_passed") is not None
                ]
            ),
            "recall_at_10": _mean(
                [
                    float(row["recall_at_10"])
                    for row in rows
                    if row.get("recall_at_10") is not None
                ]
            ),
            "fact_coverage": _mean(
                [
                    float(row["fact_coverage"])
                    for row in rows
                    if row.get("fact_coverage") is not None
                ]
            ),
        }
    return result


def _system_findings(
    diagnostics: dict[str, Any],
) -> list[dict[str, str]]:
    case_domains = {
        str(item["business_domain"])
        for item in diagnostics.get("case_domains", [])
        if item.get("business_domain")
    }
    resolved_case_domains = {
        str(domain)
        for domain in diagnostics.get("resolved_case_domains", [])
        if domain
    }
    effective_case_domains = resolved_case_domains or case_domains
    chunk_domains = {
        str(item["business_domain"])
        for item in diagnostics.get("active_chunk_domains", [])
        if item.get("business_domain")
    }
    findings = []
    retrieval_hit_count = int(
        diagnostics.get("retrieval_hit_count") or 0
    )
    if (
        retrieval_hit_count == 0
        and effective_case_domains
        and chunk_domains
        and effective_case_domains.isdisjoint(chunk_domains)
    ):
        findings.append(
            {
                "code": "BUSINESS_DOMAIN_FILTER_MISMATCH",
                "severity": "严重",
                "title": "业务域精确过滤不匹配",
                "evidence": (
                    "评测题业务域为 "
                    f"`{', '.join(sorted(case_domains))}`，"
                    "解析后的过滤业务域为 "
                    f"`{', '.join(sorted(effective_case_domains))}`，"
                    "ACTIVE Chunk 业务域为 "
                    f"`{', '.join(sorted(chunk_domains))}`，两者没有交集。"
                    "当前 BM25 和向量检索都使用业务域精确过滤，"
                    "因此候选结果会在检索阶段被全部过滤。"
                ),
                "recommendation": (
                    "统一业务域枚举，或把顶层业务域映射为允许的细分域集合；"
                    "修复后创建新的实验版本重跑，不覆盖本次 V0 基线。"
                ),
            }
        )
    return findings


def _stage_mean(
    cases: list[dict[str, Any]],
    stage: str,
) -> float | None:
    return _mean(
        [
            float(value)
            for item in cases
            if (
                value := item["attribution"]["stage_fact_coverage"].get(
                    stage
                )
            )
            is not None
        ]
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _slice_row(name: str, details: dict[str, Any]) -> str:
    return (
        f"| {_escape(name)} | {details['count']} | "
        f"{_percent(details['pass_rate'])} | "
        f"{_percent(details['judge_pass_rate'])} | "
        f"{_number(details['recall_at_10'])} | "
        f"{_number(details['fact_coverage'])} |"
    )


def _percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def _number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _escape(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _truncate(value: Any, length: int) -> str:
    text = str(value or "")
    return text if len(text) <= length else f"{text[:length - 1]}…"


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
