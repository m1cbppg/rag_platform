from types import SimpleNamespace

import pytest

from scripts.run_rag_evaluation import (
    apply_runtime_overrides,
    build_config_snapshot,
    load_experiment_cases,
    parse_dataset_reference,
)
from src.rag_platform.evaluation.models import DatasetSplit


def test_parse_dataset_reference_requires_code_and_version() -> None:
    assert parse_dataset_reference("rag_eval_ecommerce:v1") == (
        "rag_eval_ecommerce",
        "v1",
    )
    with pytest.raises(ValueError, match="code:version"):
        parse_dataset_reference("rag_eval_ecommerce")


def test_config_snapshot_contains_reproducibility_fields() -> None:
    settings = SimpleNamespace(
        es_chunk_index="rag_chunk_bm25",
        milvus_collection="rag_chunk_vector",
        embedding_model="text-embedding-v4",
        rerank_model="qwen3-rerank",
        answer_model="deepseek-chat",
        qwen_judge_model="qwen-plus",
        hybrid_fusion_method="rrf",
        rrf_rank_constant=60,
        rrf_window_size=50,
        es_bm25_top_k=20,
        hybrid_final_top_k=20,
        rerank_top_n=5,
        context_max_tokens=6000,
        context_max_chunks=8,
        query_analysis_use_llm=True,
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        adaptive_quality_good_threshold=0.68,
        adaptive_quality_poor_threshold=0.25,
        adaptive_rerank_top1_threshold=0.60,
        adaptive_rerank_top3_threshold=0.55,
        adaptive_min_candidate_count=3,
        adaptive_min_distinct_documents=2,
        adaptive_min_version_count=2,
        adaptive_rewrite_model="deepseek-chat",
        adaptive_rewrite_max_attempts=2,
        query_decomposition_enabled=True,
        query_decomposition_model="deepseek-chat",
        query_decomposition_max_sub_queries=3,
        query_decomposition_max_attempts=2,
        query_decomposition_min_query_length=18,
        query_decomposition_min_benefit_score=0.8,
        query_decomposition_allow_dependent=True,
        query_decomposition_rerank_extra_limit=3,
        sub_query_min_candidates=1,
        sub_query_rerank_quota=1,
        dependent_multi_hop_enabled=True,
        dependent_multi_hop_max_hops=2,
        dependent_fact_model="deepseek-chat",
        dependent_fact_min_confidence=0.75,
        dependent_fact_max_candidates=5,
        dependent_fact_max_attempts=2,
        action_decision_enabled=True,
        action_decision_model="deepseek-chat",
        action_decision_clarify_threshold=0.75,
        action_decision_refuse_threshold=0.8,
        action_decision_max_attempts=2,
    )
    args = SimpleNamespace(
        top_k=10,
        concurrency=1,
        split="development",
        limit=2,
        expected_actions=["refuse", "clarify"],
        case_types=["multi_hop"],
        adaptive_retrieval="enabled",
        query_decomposition="disabled",
        query_analysis="rule",
    )

    snapshot = build_config_snapshot(
        settings=settings,
        args=args,
        dataset_sha256="a" * 64,
        git_dirty=True,
    )

    assert snapshot["dataset_sha256"] == "a" * 64
    assert snapshot["elasticsearch_index"] == "rag_chunk_bm25"
    assert snapshot["milvus_collection"] == "rag_chunk_vector"
    assert snapshot["top_k"] == 10
    assert snapshot["git_dirty"] is True
    assert snapshot["business_domain_alias_version"] == "v1"
    assert snapshot["business_domain_aliases"][
        "ecommerce_after_sales"
    ] == [
        "order",
        "payment",
        "refund",
        "return",
        "after_sales",
        "logistics",
        "coupon",
        "invoice",
        "member",
        "risk",
    ]
    assert snapshot["action_decision_enabled"] is True
    assert snapshot["action_decision_model"] == "deepseek-chat"
    assert snapshot["action_decision_prompt_version"] == (
        "v8-composable-evidence-boundary"
    )
    assert snapshot["judge_prompt_version"] == "v2-action-contract"
    assert snapshot["clarification_policy_version"] == "v5"
    assert snapshot["clarification_policies"]["version"] == "v5"
    assert len(snapshot["clarification_policies"]["policies"]) == 6
    assert snapshot["evidence_constraint_guard_version"] == "v1"
    assert snapshot["action_decision_clarify_threshold"] == 0.75
    assert snapshot["action_decision_refuse_threshold"] == 0.8
    assert snapshot["action_decision_max_attempts"] == 2
    assert snapshot["adaptive_retrieval_enabled"] is True
    assert snapshot["adaptive_retrieval_cli_override"] == "enabled"
    assert snapshot["query_analysis_use_llm"] is True
    assert snapshot["query_analysis_cli_override"] == "rule"
    assert snapshot["adaptive_retrieval_policy_version"] == "v1"
    assert snapshot["adaptive_max_rounds"] == 2
    assert snapshot["adaptive_rerank_top1_threshold"] == 0.6
    assert snapshot["adaptive_rerank_top3_threshold"] == 0.55
    assert snapshot["adaptive_query_rewrite_prompt_version"] == "v1"
    assert snapshot["adaptive_rewrite_model"] == "deepseek-chat"
    assert snapshot["query_decomposition_enabled"] is True
    assert snapshot["query_decomposition_cli_override"] == "disabled"
    assert snapshot["query_decomposition_prompt_version"] == (
        "v3-dependent-two-hop"
    )
    assert snapshot["query_decomposition_max_sub_queries"] == 3
    assert snapshot["query_decomposition_min_benefit_score"] == 0.8
    assert snapshot["query_decomposition_allow_dependent"] is True
    assert snapshot["query_decomposition_rerank_extra_limit"] == 3
    assert snapshot["sub_query_rerank_quota"] == 1
    assert snapshot["dependent_multi_hop_enabled"] is True
    assert snapshot["dependent_multi_hop_max_hops"] == 2
    assert snapshot["dependent_fact_min_confidence"] == 0.75
    assert snapshot["dependent_fact_prompt_version"] == (
        "v2-first-hop-answer-bound"
    )
    assert snapshot["expected_actions"] == ["REFUSE", "CLARIFY"]
    assert snapshot["case_codes"] == []
    assert snapshot["case_types"] == ["MULTI_HOP"]


def test_runtime_override_can_disable_adaptive_retrieval() -> None:
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        query_decomposition_enabled=True,
        query_analysis_use_llm=True,
    )
    args = SimpleNamespace(
        adaptive_retrieval="disabled",
        query_decomposition="disabled",
        query_analysis="rule",
    )

    result = apply_runtime_overrides(settings, args)

    assert result is settings
    assert settings.adaptive_retrieval_enabled is False
    assert settings.query_decomposition_enabled is False
    assert settings.query_analysis_use_llm is False


class FakeDatasetRepository:
    def list_reviewed_cases(self, dataset_id, split):
        assert dataset_id == 7
        assert split == DatasetSplit.DEVELOPMENT
        return [
            {
                "id": 11,
                "case_code": "CASE_DIRECT_001",
                "question": "待支付订单如何取消？",
                "reference_answer": "可以直接取消。",
                "case_type": "DIRECT",
                "target_doc_types": ["FAQ"],
                "expected_action": "ANSWER",
                "difficulty": "EASY",
                "dataset_split": "DEVELOPMENT",
                "business_domain": "ecommerce_after_sales",
                "required_fact_count": 1,
                "generation_metadata": {},
                "review_status": "PASSED",
                "review_score": 0.95,
                "review_reason": "通过",
                "status": "ACTIVE",
            }
        ]

    def list_case_evidence(self, case_id):
        assert case_id == 11
        return [
            {
                "source_doc_code": "FAQ_ORDER_STATUS_001",
                "evidence_quote": "待支付订单可以直接取消。",
                "fact_key": "cancel_pending",
                "relevance_grade": 3,
                "mapped_doc_id": 1,
                "mapped_chunk_id": 101,
                "mapping_status": "MAPPED",
                "mapping_reason": "唯一命中",
            }
        ]


def test_load_experiment_cases_builds_reviewed_models() -> None:
    cases = load_experiment_cases(
        repository=FakeDatasetRepository(),
        dataset_id=7,
        split=DatasetSplit.DEVELOPMENT,
        limit=1,
    )

    assert len(cases) == 1
    assert cases[0].case_id == 11
    assert cases[0].case.evidences[0].mapped_chunk_id == 101


class FakeActionFilterRepository(FakeDatasetRepository):
    def list_reviewed_cases(self, dataset_id, split):
        answer = super().list_reviewed_cases(dataset_id, split)[0]
        return [
            answer,
            {
                **answer,
                "id": 12,
                "case_code": "CASE_NO_ANSWER_001",
                "question": "天气怎么样？",
                "reference_answer": None,
                "case_type": "NO_ANSWER",
                "expected_action": "REFUSE",
                "required_fact_count": 0,
            },
            {
                **answer,
                "id": 13,
                "case_code": "CASE_NO_ANSWER_002",
                "question": "商品坏了怎么办？",
                "reference_answer": None,
                "case_type": "NO_ANSWER",
                "expected_action": "CLARIFY",
                "required_fact_count": 0,
            },
        ]

    def list_case_evidence(self, case_id):
        if case_id in (12, 13):
            return []
        return super().list_case_evidence(case_id)


def test_load_experiment_cases_filters_expected_actions_before_limit() -> None:
    cases = load_experiment_cases(
        repository=FakeActionFilterRepository(),
        dataset_id=7,
        split=DatasetSplit.DEVELOPMENT,
        expected_actions=["refuse", "clarify"],
        limit=2,
    )

    assert [item.case.expected_action.value for item in cases] == [
        "REFUSE",
        "CLARIFY",
    ]


def test_load_experiment_cases_filters_case_codes() -> None:
    cases = load_experiment_cases(
        repository=FakeActionFilterRepository(),
        dataset_id=7,
        split=DatasetSplit.DEVELOPMENT,
        case_codes=["CASE_NO_ANSWER_002"],
    )

    assert [item.case.case_code for item in cases] == [
        "CASE_NO_ANSWER_002"
    ]


def test_load_experiment_cases_filters_case_types() -> None:
    cases = load_experiment_cases(
        repository=FakeActionFilterRepository(),
        dataset_id=7,
        split=DatasetSplit.DEVELOPMENT,
        case_types=["direct", "no_answer"],
    )

    assert [item.case.case_type.value for item in cases] == [
        "DIRECT",
        "NO_ANSWER",
        "NO_ANSWER",
    ]
