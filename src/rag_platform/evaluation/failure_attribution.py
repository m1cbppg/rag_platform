from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AttributionCode(StrEnum):
    PASS = "PASS"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    RETRIEVAL_PARTIAL = "RETRIEVAL_PARTIAL"
    RERANK_DROPPED = "RERANK_DROPPED"
    RERANK_PARTIAL_DROP = "RERANK_PARTIAL_DROP"
    CONTEXT_DROPPED = "CONTEXT_DROPPED"
    CONTEXT_PARTIAL_DROP = "CONTEXT_PARTIAL_DROP"
    FALSE_REFUSAL = "FALSE_REFUSAL"
    FALSE_ANSWER = "FALSE_ANSWER"
    ACTION_MISMATCH = "ACTION_MISMATCH"
    CLARIFICATION_NOT_SUPPORTED = "CLARIFICATION_NOT_SUPPORTED"
    REFUSAL_QUALITY_FAILURE = "REFUSAL_QUALITY_FAILURE"
    CLARIFICATION_QUALITY_FAILURE = "CLARIFICATION_QUALITY_FAILURE"
    CITATION_FAILURE = "CITATION_FAILURE"
    ANSWER_INCOMPLETE = "ANSWER_INCOMPLETE"
    ANSWER_UNFAITHFUL = "ANSWER_UNFAITHFUL"
    ANSWER_IRRELEVANT = "ANSWER_IRRELEVANT"
    CONFLICT_HANDLING_FAILURE = "CONFLICT_HANDLING_FAILURE"
    JUDGE_MISSING = "JUDGE_MISSING"
    ANSWER_QUALITY_FAILURE = "ANSWER_QUALITY_FAILURE"


_LABELS = {
    AttributionCode.PASS: "通过",
    AttributionCode.EXECUTION_ERROR: "执行异常",
    AttributionCode.RETRIEVAL_MISS: "融合召回完全缺失",
    AttributionCode.RETRIEVAL_PARTIAL: "融合召回部分缺失",
    AttributionCode.RERANK_DROPPED: "精排完全淘汰标准证据",
    AttributionCode.RERANK_PARTIAL_DROP: "精排部分淘汰必要事实",
    AttributionCode.CONTEXT_DROPPED: "Context 完全丢失标准证据",
    AttributionCode.CONTEXT_PARTIAL_DROP: "Context 部分丢失必要事实",
    AttributionCode.FALSE_REFUSAL: "有证据但系统错误拒答",
    AttributionCode.FALSE_ANSWER: "应拒答但系统错误回答",
    AttributionCode.ACTION_MISMATCH: "系统行为与预期不一致",
    AttributionCode.CLARIFICATION_NOT_SUPPORTED: "系统缺少澄清行为",
    AttributionCode.REFUSAL_QUALITY_FAILURE: "拒答质量未通过",
    AttributionCode.CLARIFICATION_QUALITY_FAILURE: "澄清质量未通过",
    AttributionCode.CITATION_FAILURE: "引用未覆盖或不能支持结论",
    AttributionCode.ANSWER_INCOMPLETE: "答案遗漏必要事实",
    AttributionCode.ANSWER_UNFAITHFUL: "答案包含上下文不支持的内容",
    AttributionCode.ANSWER_IRRELEVANT: "答案与问题不相关",
    AttributionCode.CONFLICT_HANDLING_FAILURE: "冲突版本处理错误",
    AttributionCode.JUDGE_MISSING: "缺少 Judge 结果",
    AttributionCode.ANSWER_QUALITY_FAILURE: "答案总体质量未通过",
}

_RECOMMENDATIONS = {
    AttributionCode.PASS: "保持当前链路，后续用于回归对照。",
    AttributionCode.EXECUTION_ERROR: "先排查外部服务、超时和数据写入错误。",
    AttributionCode.RETRIEVAL_MISS: (
        "优先检查 Query 改写、检索路由、过滤条件、索引内容和召回 Top K。"
    ),
    AttributionCode.RETRIEVAL_PARTIAL: (
        "增加多事实查询拆分或候选池，检查多跳事实是否被单次查询覆盖。"
    ),
    AttributionCode.RERANK_DROPPED: (
        "检查 rerank candidate limit、top_n、最低分阈值和精排文本构造。"
    ),
    AttributionCode.RERANK_PARTIAL_DROP: (
        "提高精排保留数量，并针对多事实题调整精排目标。"
    ),
    AttributionCode.CONTEXT_DROPPED: (
        "检查 Context Chunk 上限、Token 预算、扩展和去重排序。"
    ),
    AttributionCode.CONTEXT_PARTIAL_DROP: (
        "为多事实题预留 Context 配额，避免同类 Chunk 占满预算。"
    ),
    AttributionCode.FALSE_REFUSAL: "检查空 Context 判断和拒答触发条件。",
    AttributionCode.FALSE_ANSWER: "增加无答案识别和回答前证据充分性判断。",
    AttributionCode.ACTION_MISMATCH: "检查行为路由和答案前置决策逻辑。",
    AttributionCode.CLARIFICATION_NOT_SUPPORTED: (
        "把 Query 分析中的 need_clarification 接入 Chat 行为输出。"
    ),
    AttributionCode.REFUSAL_QUALITY_FAILURE: "调整拒答模板和 Judge 契约。",
    AttributionCode.CLARIFICATION_QUALITY_FAILURE: "优化澄清问题的具体性。",
    AttributionCode.CITATION_FAILURE: (
        "强化引用约束，并校验每个必要事实是否有对应引用。"
    ),
    AttributionCode.ANSWER_INCOMPLETE: (
        "在答案 Prompt 中显式要求覆盖全部必要事实。"
    ),
    AttributionCode.ANSWER_UNFAITHFUL: (
        "收紧基于 Context 回答的约束并降低无证据扩写。"
    ),
    AttributionCode.ANSWER_IRRELEVANT: "优化问题改写和答案 Prompt 的任务约束。",
    AttributionCode.CONFLICT_HANDLING_FAILURE: (
        "加入版本、生效时间和当前有效规则的冲突消解策略。"
    ),
    AttributionCode.JUDGE_MISSING: "补跑 Judge，确认评审服务和结果写入状态。",
    AttributionCode.ANSWER_QUALITY_FAILURE: "检查 Judge 理由后细化答案生成策略。",
}


def attribution_label(code: AttributionCode | str) -> str:
    return _LABELS[AttributionCode(code)]


def attribution_recommendation(code: AttributionCode | str) -> str:
    return _RECOMMENDATIONS[AttributionCode(code)]


class CaseAttribution(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    case_code: str
    primary_code: AttributionCode
    primary_label: str
    secondary_codes: list[AttributionCode] = Field(default_factory=list)
    stage_fact_coverage: dict[str, float | None]
    stage_gold_chunks: dict[str, list[int]]
    missing_fact_keys: dict[str, list[str]]
    basis: list[str] = Field(default_factory=list)
    recommendation: str

    @property
    def passed(self) -> bool:
        return self.primary_code == AttributionCode.PASS


def attribute_case(
    *,
    case_result: dict[str, Any],
    hits: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    judge_threshold: float = 0.8,
) -> CaseAttribution:
    fact_keys_by_chunk = _fact_keys_by_chunk(evidences)
    all_fact_keys = {
        fact_key
        for fact_keys in fact_keys_by_chunk.values()
        for fact_key in fact_keys
    }
    stage_chunks = _stage_chunks(hits)
    stage_fact_coverage = {
        stage: _fact_coverage(
            chunk_ids=chunk_ids,
            fact_keys_by_chunk=fact_keys_by_chunk,
            all_fact_keys=all_fact_keys,
        )
        for stage, chunk_ids in stage_chunks.items()
    }
    missing_fact_keys = {
        stage: sorted(
            all_fact_keys
            - _covered_fact_keys(chunk_ids, fact_keys_by_chunk)
        )
        for stage, chunk_ids in stage_chunks.items()
    }
    secondary = _answer_quality_failures(
        case_result,
        threshold=judge_threshold,
    )
    primary, basis = _primary_attribution(
        case_result=case_result,
        stage_fact_coverage=stage_fact_coverage,
        secondary=secondary,
    )
    secondary = [code for code in secondary if code != primary]
    return CaseAttribution(
        case_code=str(case_result.get("case_code") or ""),
        primary_code=primary,
        primary_label=_LABELS[primary],
        secondary_codes=secondary,
        stage_fact_coverage=stage_fact_coverage,
        stage_gold_chunks={
            stage: sorted(chunk_ids)
            for stage, chunk_ids in stage_chunks.items()
        },
        missing_fact_keys=missing_fact_keys,
        basis=basis,
        recommendation=_RECOMMENDATIONS[primary],
    )


def _primary_attribution(
    *,
    case_result: dict[str, Any],
    stage_fact_coverage: dict[str, float | None],
    secondary: list[AttributionCode],
) -> tuple[AttributionCode, list[str]]:
    status = case_result.get("status")
    actual_action = case_result.get("actual_action")
    expected_action = case_result.get("expected_action")
    if status == "FAILED" or actual_action == "ERROR":
        return AttributionCode.EXECUTION_ERROR, [
            str(case_result.get("error_message") or "逐题执行失败")
        ]

    if expected_action == "REFUSE":
        if actual_action != "REFUSE":
            code = (
                AttributionCode.FALSE_ANSWER
                if actual_action == "ANSWER"
                else AttributionCode.ACTION_MISMATCH
            )
            return code, [f"预期 REFUSE，实际 {actual_action}"]
        if case_result.get("judge_passed") in (1, True):
            return AttributionCode.PASS, ["拒答行为和 Judge 均通过"]
        return AttributionCode.REFUSAL_QUALITY_FAILURE, [
            "行为为 REFUSE，但 Judge 未通过"
        ]

    if expected_action == "CLARIFY":
        if actual_action != "CLARIFY":
            return AttributionCode.CLARIFICATION_NOT_SUPPORTED, [
                f"预期 CLARIFY，实际 {actual_action}"
            ]
        if case_result.get("judge_passed") in (1, True):
            return AttributionCode.PASS, ["澄清行为和 Judge 均通过"]
        return AttributionCode.CLARIFICATION_QUALITY_FAILURE, [
            "行为为 CLARIFY，但 Judge 未通过"
        ]

    merged = stage_fact_coverage["merged"]
    rerank = stage_fact_coverage["rerank"]
    final = stage_fact_coverage["final"]
    if merged == 0:
        return AttributionCode.RETRIEVAL_MISS, [
            "融合召回阶段没有覆盖任何必要事实"
        ]
    if merged is not None and merged < 1:
        return AttributionCode.RETRIEVAL_PARTIAL, [
            f"融合召回必要事实覆盖率为 {merged:.4f}"
        ]
    if rerank == 0:
        return AttributionCode.RERANK_DROPPED, [
            "融合召回已覆盖必要事实，但精排结果覆盖率为 0"
        ]
    if (
        rerank is not None
        and merged is not None
        and rerank < merged
    ):
        return AttributionCode.RERANK_PARTIAL_DROP, [
            f"精排覆盖率从 {merged:.4f} 降至 {rerank:.4f}"
        ]
    if final == 0:
        return AttributionCode.CONTEXT_DROPPED, [
            "精排已覆盖必要事实，但最终 Context 覆盖率为 0"
        ]
    if final is not None and rerank is not None and final < rerank:
        return AttributionCode.CONTEXT_PARTIAL_DROP, [
            f"Context 覆盖率从 {rerank:.4f} 降至 {final:.4f}"
        ]
    if actual_action == "REFUSE":
        return AttributionCode.FALSE_REFUSAL, [
            "必要事实已进入 Context，但系统返回 REFUSE"
        ]
    if actual_action != "ANSWER":
        return AttributionCode.ACTION_MISMATCH, [
            f"预期 ANSWER，实际 {actual_action}"
        ]
    if secondary:
        return secondary[0], ["必要事实已进入 Context，答案质量指标未通过"]
    if case_result.get("judge_passed") is None:
        return AttributionCode.JUDGE_MISSING, ["逐题结果缺少 Judge 记录"]
    if case_result.get("judge_passed") not in (1, True):
        return AttributionCode.ANSWER_QUALITY_FAILURE, [
            "Judge 未通过，但没有命中更具体的低分维度"
        ]
    return AttributionCode.PASS, ["检索、行为、引用和 Judge 均通过"]


def _answer_quality_failures(
    case_result: dict[str, Any],
    *,
    threshold: float,
) -> list[AttributionCode]:
    failures: list[AttributionCode] = []
    citation_recall = case_result.get("citation_recall")
    citation_score = case_result.get("citation_entailment_score")
    if (
        citation_recall is not None
        and float(citation_recall) < 1
    ) or _below(citation_score, threshold):
        failures.append(AttributionCode.CITATION_FAILURE)
    if _below(case_result.get("completeness_score"), threshold):
        failures.append(AttributionCode.ANSWER_INCOMPLETE)
    if _below(case_result.get("faithfulness_score"), threshold):
        failures.append(AttributionCode.ANSWER_UNFAITHFUL)
    if _below(case_result.get("answer_relevance_score"), threshold):
        failures.append(AttributionCode.ANSWER_IRRELEVANT)
    if (
        case_result.get("case_type") == "CONFLICT"
        and _below(
            case_result.get("conflict_handling_score"),
            threshold,
        )
    ):
        failures.append(AttributionCode.CONFLICT_HANDLING_FAILURE)
    return failures


def _fact_keys_by_chunk(
    evidences: list[dict[str, Any]],
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for evidence in evidences:
        chunk_id = evidence.get("mapped_chunk_id")
        fact_key = evidence.get("fact_key")
        if (
            evidence.get("mapping_status") != "MAPPED"
            or chunk_id is None
            or not fact_key
        ):
            continue
        result.setdefault(int(chunk_id), set()).add(str(fact_key))
    return result


def _stage_chunks(
    hits: list[dict[str, Any]],
) -> dict[str, set[int]]:
    stages = {
        "merged": set(),
        "rerank": set(),
        "final": set(),
    }
    for hit in hits:
        channel = str(hit.get("channel") or "").upper()
        chunk_id = hit.get("chunk_id")
        if chunk_id is None:
            continue
        if channel == "RERANK":
            stages["rerank"].add(int(chunk_id))
        elif channel == "FINAL":
            stages["final"].add(int(chunk_id))
        else:
            stages["merged"].add(int(chunk_id))
    return stages


def _fact_coverage(
    *,
    chunk_ids: set[int],
    fact_keys_by_chunk: dict[int, set[str]],
    all_fact_keys: set[str],
) -> float | None:
    if not all_fact_keys:
        return None
    return len(
        _covered_fact_keys(chunk_ids, fact_keys_by_chunk)
    ) / len(all_fact_keys)


def _covered_fact_keys(
    chunk_ids: set[int],
    fact_keys_by_chunk: dict[int, set[str]],
) -> set[str]:
    return {
        fact_key
        for chunk_id in chunk_ids
        for fact_key in fact_keys_by_chunk.get(chunk_id, set())
    }


def _below(value: Any, threshold: float) -> bool:
    return value is not None and float(value) < threshold
