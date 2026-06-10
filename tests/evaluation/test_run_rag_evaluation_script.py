from types import SimpleNamespace

import pytest

from scripts.run_rag_evaluation import (
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
    )
    args = SimpleNamespace(
        top_k=10,
        concurrency=1,
        split="development",
        limit=2,
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
