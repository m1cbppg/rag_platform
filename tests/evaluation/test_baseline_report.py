from src.rag_platform.evaluation.baseline_report import (
    build_baseline_report,
    render_markdown,
)


def _case_result(
    *,
    result_id: int,
    case_id: int,
    case_code: str,
    case_type: str,
    difficulty: str,
    question: str,
    judge_passed: int,
    recall_at_10: float,
    fact_coverage: float,
) -> dict:
    return {
        "id": result_id,
        "case_id": case_id,
        "case_code": case_code,
        "question": question,
        "reference_answer": "参考答案",
        "case_type": case_type,
        "difficulty": difficulty,
        "expected_action": "ANSWER",
        "actual_action": "ANSWER",
        "status": "SUCCESS",
        "action_correct": 1,
        "recall_at_10": recall_at_10,
        "fact_coverage": fact_coverage,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "judge_passed": judge_passed,
        "faithfulness_score": 1.0,
        "answer_relevance_score": 1.0,
        "completeness_score": 1.0 if judge_passed else 0.4,
        "citation_entailment_score": 1.0,
        "conflict_handling_score": None,
        "latency_ms": 100,
        "error_message": None,
        "judge_reason": {},
    }


def test_builds_attribution_summary_funnel_and_priorities() -> None:
    cases = [
        _case_result(
            result_id=11,
            case_id=1,
            case_code="CASE_PASS",
            case_type="DIRECT",
            difficulty="EASY",
            question="成功题",
            judge_passed=1,
            recall_at_10=1.0,
            fact_coverage=1.0,
        ),
        _case_result(
            result_id=12,
            case_id=2,
            case_code="CASE_MISS",
            case_type="MULTI_HOP",
            difficulty="HARD",
            question="召回失败题",
            judge_passed=0,
            recall_at_10=0.0,
            fact_coverage=0.0,
        ),
    ]
    hits = [
        {
            "case_result_id": 11,
            "channel": "HYBRID",
            "chunk_id": 101,
            "rank_no": 1,
        },
        {
            "case_result_id": 11,
            "channel": "RERANK",
            "chunk_id": 101,
            "rank_no": 1,
        },
        {
            "case_result_id": 11,
            "channel": "FINAL",
            "chunk_id": 101,
            "rank_no": 1,
        },
    ]
    evidences = [
        {
            "case_id": 1,
            "mapped_chunk_id": 101,
            "fact_key": "fact_a",
            "mapping_status": "MAPPED",
        },
        {
            "case_id": 2,
            "mapped_chunk_id": 202,
            "fact_key": "fact_b",
            "mapping_status": "MAPPED",
        },
    ]

    report = build_baseline_report(
        run={
            "id": 7,
            "run_code": "V0_DEV",
            "experiment_version": "V0",
            "experiment_name": "baseline",
            "status": "SUCCESS",
            "total_cases": 2,
            "completed_cases": 2,
            "failed_cases": 0,
            "config": {"top_k": 20},
            "summary_metrics": {},
        },
        case_results=cases,
        hits=hits,
        evidences=evidences,
        system_diagnostics={
            "retrieval_hit_count": 3,
            "case_domains": [
                {
                    "business_domain": "ecommerce_after_sales",
                    "case_count": 2,
                }
            ],
            "active_chunk_domains": [
                {"business_domain": "order", "chunk_count": 30}
            ],
        },
    )

    assert report["overview"]["attribution_pass_rate"] == 0.5
    assert report["attribution_summary"]["PASS"]["count"] == 1
    assert report["attribution_summary"]["RETRIEVAL_MISS"]["count"] == 1
    assert report["stage_funnel"]["merged_fact_coverage"] == 0.5
    assert report["stage_funnel"]["final_fact_coverage"] == 0.5
    assert report["by_case_type"]["MULTI_HOP"]["pass_rate"] == 0.0
    assert report["by_difficulty"]["EASY"]["pass_rate"] == 1.0
    assert report["priorities"][0]["code"] == "RETRIEVAL_MISS"
    assert report["cases"][1]["attribution"]["primary_code"] == (
        "RETRIEVAL_MISS"
    )
    assert report["system_findings"] == []


def test_renders_chinese_markdown_with_failed_case_table() -> None:
    report = build_baseline_report(
        run={
            "id": 7,
            "run_code": "V0_DEV",
            "experiment_version": "V0",
            "experiment_name": "baseline",
            "status": "SUCCESS",
            "total_cases": 1,
            "completed_cases": 1,
            "failed_cases": 0,
            "config": {"top_k": 20},
            "summary_metrics": {},
        },
        case_results=[
            _case_result(
                result_id=12,
                case_id=2,
                case_code="CASE_MISS",
                case_type="MULTI_HOP",
                difficulty="HARD",
                question="召回失败题",
                judge_passed=0,
                recall_at_10=0.0,
                fact_coverage=0.0,
            )
        ],
        hits=[],
        evidences=[
            {
                "case_id": 2,
                "mapped_chunk_id": 202,
                "fact_key": "fact_b",
                "mapping_status": "MAPPED",
            }
        ],
        system_diagnostics={
            "retrieval_hit_count": 0,
            "case_domains": [
                {
                    "business_domain": "ecommerce_after_sales",
                    "case_count": 1,
                }
            ],
            "active_chunk_domains": [
                {"business_domain": "refund", "chunk_count": 30}
            ],
        },
    )

    markdown = render_markdown(report)

    assert "# V0_DEV RAG 基线评测报告" in markdown
    assert "## 失败归因分布" in markdown
    assert "融合召回完全缺失" in markdown
    assert "召回失败题" in markdown
    assert "业务域精确过滤不匹配" in markdown
    assert "## V0 后续优化优先级" in markdown
