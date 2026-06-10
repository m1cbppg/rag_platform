import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from src.rag_platform.evaluation.dataset_repository import DatasetRepository
from src.rag_platform.evaluation.models import (
    ActualAction,
    DatasetSplit,
    DatasetStatus,
    EvalCaseStatus,
    EvalCaseType,
    EvalRunConfig,
    EvalRunStatus,
    EvidenceSpec,
    ExpectedAction,
    GeneratedEvalCase,
    JudgeScore,
    MappingStatus,
    RetrievalMetricResult,
    ReviewStatus,
    SourceDocumentSpec,
    SourceDocumentType,
)
from src.rag_platform.infrastructure.mysql import create_mysql_engine


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_MYSQL_INTEGRATION") != "1",
        reason="设置 RUN_MYSQL_INTEGRATION=1 后才连接真实 MySQL",
    ),
]


def test_repository_persists_complete_evaluation_lifecycle() -> None:
    engine = create_mysql_engine()
    repository = DatasetRepository(engine=engine)
    suffix = uuid4().hex[:10]
    dataset_id: int | None = None
    run_id: int | None = None

    try:
        dataset_id = repository.create_dataset(
            dataset_code=f"TEST_EVAL_{suffix}",
            name="M1 Repository 集成测试",
            version="v1",
            domain="ecommerce_after_sales",
            description="测试结束后自动删除",
            generator_provider="deepseek",
            generator_model="deepseek-chat",
            reviewer_provider="dashscope",
            reviewer_model="qwen-plus",
            generation_config={"purpose": "integration-test"},
        )

        source_document_id = repository.save_source_document(
            dataset_id=dataset_id,
            document=SourceDocumentSpec(
                source_doc_code=f"RULE_REFUND_{suffix}",
                title="退款规则",
                doc_type=SourceDocumentType.RULE,
                topic="refund",
                version="v1",
                relative_file_path=f"rules/refund-{suffix}.docx",
                source_content_sha256="a" * 64,
                generation_spec={"required_fact": "refund_rule"},
            ),
        )
        repository.update_source_document_review(
            source_document_id=source_document_id,
            review_status=ReviewStatus.PASSED,
            review_score=0.95,
            review_reason="审核通过",
        )
        repository.map_source_document(
            dataset_id=dataset_id,
            source_doc_code=f"RULE_REFUND_{suffix}",
            mapped_doc_id=900001,
        )
        repository.map_source_document(
            dataset_id=dataset_id,
            source_doc_code=f"RULE_REFUND_{suffix}",
            mapped_doc_id=900001,
        )

        evidence = EvidenceSpec(
            source_doc_code=f"RULE_REFUND_{suffix}",
            evidence_quote="未发货订单允许直接申请退款。",
            fact_key="refund_rule",
            relevance_grade=3,
            mapping_status=MappingStatus.PENDING,
        )
        case_id = repository.save_eval_case(
            dataset_id=dataset_id,
            case=GeneratedEvalCase(
                case_code=f"CASE_{suffix}",
                question="未发货订单可以退款吗？",
                normalized_question="未发货订单可以退款吗",
                reference_answer="未发货订单允许直接申请退款。",
                case_type=EvalCaseType.DIRECT,
                target_doc_types=[SourceDocumentType.RULE],
                expected_action=ExpectedAction.ANSWER,
                dataset_split=DatasetSplit.DEVELOPMENT,
                business_domain="ecommerce_after_sales",
                required_fact_count=1,
                generation_metadata={"source_group": "refund-v1"},
                evidences=[evidence],
            ),
        )
        upserted_case_id = repository.upsert_eval_case(
            dataset_id=dataset_id,
            case=GeneratedEvalCase(
                case_code=f"CASE_{suffix}",
                question="待支付订单是否允许退款？",
                normalized_question="待支付订单是否允许退款",
                reference_answer="未发货订单允许直接申请退款。",
                case_type=EvalCaseType.DIRECT,
                target_doc_types=[SourceDocumentType.RULE],
                expected_action=ExpectedAction.ANSWER,
                dataset_split=DatasetSplit.DEVELOPMENT,
                business_domain="ecommerce_after_sales",
                required_fact_count=1,
                generation_metadata={"source_group": "refund-v1"},
                evidences=[evidence],
            ),
        )
        assert upserted_case_id == case_id
        repository.update_eval_case_review(
            case_id=case_id,
            review_status=ReviewStatus.PASSED,
            review_score=0.95,
            review_reason="审核通过",
            status=EvalCaseStatus.ACTIVE,
        )
        evidence_id = repository.save_case_evidence(
            case_id=case_id,
            source_document_id=source_document_id,
            evidence=evidence,
        )
        repository.map_case_evidence(
            evidence_id=evidence_id,
            mapped_doc_id=900001,
            mapped_chunk_id=910001,
        )
        repository.map_case_evidence(
            evidence_id=evidence_id,
            mapped_doc_id=900001,
            mapped_chunk_id=910001,
        )

        reviewed_cases = repository.list_reviewed_cases(
            dataset_id=dataset_id,
            split=DatasetSplit.DEVELOPMENT,
        )
        case_evidences = repository.list_case_evidence(case_id)
        assert len(reviewed_cases) == 1
        assert reviewed_cases[0]["target_doc_types"] == ["RULE"]
        assert case_evidences[0]["mapping_status"] == "MAPPED"
        assert case_evidences[0]["mapped_chunk_id"] == 910001

        repository.freeze_dataset(
            dataset_id=dataset_id,
            content_sha256="b" * 64,
        )

        run_id = repository.create_run(
            EvalRunConfig(
                run_code=f"RUN_{suffix}",
                dataset_id=dataset_id,
                experiment_version="V0",
                experiment_name="M1 Repository 基线测试",
                git_commit_sha="test-commit",
                retrieval_mode="hybrid",
                embedding_model="text-embedding-v4",
                rerank_model="qwen3-rerank",
                answer_model="deepseek-chat",
                judge_model="qwen-plus",
                config={"top_k": 10, "rerank_top_n": 5},
                total_cases=1,
            )
        )
        repository.start_run(run_id)

        case_result_id = repository.start_case_result(
            run_id=run_id,
            case_id=case_id,
            trace_id=f"trace-{suffix}",
        )
        repository.save_retrieval_hits(
            case_result_id=case_result_id,
            hits=[
                {
                    "retrieval_round": 1,
                    "query_variant": "ORIGINAL",
                    "query_text": "未发货订单可以退款吗？",
                    "channel": "FINAL",
                    "chunk_id": 910001,
                    "rank_no": 1,
                    "raw_score": 0.9,
                    "fused_score": 0.8,
                    "rerank_score": 0.95,
                    "is_gold": True,
                    "metadata": {"source": "integration-test"},
                }
            ],
        )
        repository.finish_case_result(
            case_result_id=case_result_id,
            actual_action=ActualAction.ANSWER,
            generated_answer="未发货订单允许直接申请退款。[C1]",
            retrieved_chunk_ids=[910001],
            cited_chunk_ids=[910001],
            metrics=RetrievalMetricResult(
                recall_at_1=1.0,
                recall_at_3=1.0,
                recall_at_5=1.0,
                recall_at_10=1.0,
                reciprocal_rank=1.0,
                ndcg_at_5=1.0,
                ndcg_at_10=1.0,
                citation_precision=1.0,
                citation_recall=1.0,
                action_correct=True,
                retrieval_rounds=1,
            ),
            input_tokens=100,
            output_tokens=20,
            estimated_cost=0.001,
            latency_ms=120,
        )
        judge_result_id = repository.save_judge_result(
            case_result_id=case_result_id,
            score=JudgeScore(
                judge_provider="dashscope",
                judge_model="qwen-plus",
                judge_prompt_version="v1",
                faithfulness_score=1.0,
                answer_relevance_score=1.0,
                completeness_score=1.0,
                citation_entailment_score=1.0,
                conflict_handling_score=1.0,
                passed=True,
                reason={"summary": "全部通过"},
                raw_response={"result": "pass"},
                latency_ms=30,
            ),
        )
        repository.finish_run(
            run_id=run_id,
            status=EvalRunStatus.SUCCESS,
            completed_cases=1,
            failed_cases=0,
            summary_metrics={"recall_at_5": 1.0},
        )

        with engine.begin() as connection:
            result_row = connection.execute(
                text(
                    """
                    SELECT status, recall_at_5, generated_answer
                    FROM rag_eval_case_result
                    WHERE id = :id
                    """
                ),
                {"id": case_result_id},
            ).mappings().one()
            run_row = connection.execute(
                text(
                    """
                    SELECT status, completed_cases, summary_metrics_json
                    FROM rag_eval_run
                    WHERE id = :id
                    """
                ),
                {"id": run_id},
            ).mappings().one()
            dataset_row = connection.execute(
                text(
                    """
                    SELECT status, content_sha256, frozen_at
                    FROM rag_eval_dataset
                    WHERE id = :id
                    """
                ),
                {"id": dataset_id},
            ).mappings().one()

        assert source_document_id > 0
        assert evidence_id > 0
        assert judge_result_id > 0
        assert result_row["status"] == "SUCCESS"
        assert float(result_row["recall_at_5"]) == 1.0
        assert result_row["generated_answer"].endswith("[C1]")
        assert run_row["status"] == "SUCCESS"
        assert run_row["completed_cases"] == 1
        assert dataset_row["status"] == DatasetStatus.FROZEN.value
        assert dataset_row["content_sha256"] == "b" * 64
        assert dataset_row["frozen_at"] is not None

    finally:
        if run_id is not None:
            repository.delete_run(run_id)
        if dataset_id is not None:
            repository.delete_dataset(dataset_id)
        engine.dispose()
