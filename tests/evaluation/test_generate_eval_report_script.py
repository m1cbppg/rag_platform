import json

import pytest

from scripts.generate_eval_report import (
    generate_report,
    validate_run_code,
)


class FakeRepository:
    def find_run_by_code(self, run_code):
        if run_code != "V0_DEV":
            return None
        return {
            "id": 7,
            "run_code": run_code,
            "experiment_version": "V0",
            "experiment_name": "baseline",
            "status": "SUCCESS",
            "total_cases": 1,
            "completed_cases": 1,
            "failed_cases": 0,
            "config": {"top_k": 20},
            "summary_metrics": {},
        }

    def list_run_case_results(self, run_id):
        assert run_id == 7
        return [
            {
                "id": 11,
                "case_id": 1,
                "case_code": "CASE_PASS",
                "question": "成功题",
                "reference_answer": "参考答案",
                "case_type": "DIRECT",
                "difficulty": "EASY",
                "expected_action": "ANSWER",
                "actual_action": "ANSWER",
                "status": "SUCCESS",
                "action_correct": 1,
                "recall_at_10": 1.0,
                "fact_coverage": 1.0,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "judge_passed": 1,
                "faithfulness_score": 1.0,
                "answer_relevance_score": 1.0,
                "completeness_score": 1.0,
                "citation_entailment_score": 1.0,
                "conflict_handling_score": None,
                "latency_ms": 100,
                "error_message": None,
                "judge_reason": {},
            }
        ]

    def list_run_retrieval_hits(self, run_id):
        return [
            {
                "case_result_id": 11,
                "channel": channel,
                "chunk_id": 101,
                "rank_no": 1,
                "metadata": {},
            }
            for channel in ("HYBRID", "RERANK", "FINAL")
        ]

    def list_run_evidences(self, run_id):
        return [
            {
                "case_id": 1,
                "mapped_chunk_id": 101,
                "fact_key": "fact_a",
                "mapping_status": "MAPPED",
            }
        ]

    def get_run_domain_diagnostics(self, run_id):
        return {
            "retrieval_hit_count": 3,
            "case_domains": [
                {
                    "business_domain": "ecommerce_after_sales",
                    "case_count": 1,
                }
            ],
            "active_chunk_domains": [
                {"business_domain": "order", "chunk_count": 30}
            ],
        }


def test_validate_run_code_rejects_path_traversal() -> None:
    assert validate_run_code("V0_DEV-001") == "V0_DEV-001"
    with pytest.raises(ValueError, match="run_code"):
        validate_run_code("../V0_DEV")


def test_generate_report_writes_json_and_chinese_markdown(tmp_path) -> None:
    result = generate_report(
        repository=FakeRepository(),
        run_code="V0_DEV",
        output_dir=tmp_path,
    )

    assert result["json_path"] == tmp_path / "V0_DEV.json"
    assert result["markdown_path"] == tmp_path / "V0_DEV.md"
    payload = json.loads(result["json_path"].read_text(encoding="utf-8"))
    markdown = result["markdown_path"].read_text(encoding="utf-8")
    assert payload["overview"]["attribution_pass_rate"] == 1.0
    assert payload["cases"][0]["attribution"]["primary_code"] == "PASS"
    assert "# V0_DEV RAG 基线评测报告" in markdown


def test_generate_report_requires_existing_run(tmp_path) -> None:
    with pytest.raises(ValueError, match="不存在"):
        generate_report(
            repository=FakeRepository(),
            run_code="MISSING",
            output_dir=tmp_path,
        )
