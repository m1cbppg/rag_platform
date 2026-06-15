from types import SimpleNamespace

import pytest

from scripts.create_corrected_eval_dataset import (
    create_corrected_dataset,
)
from src.rag_platform.evaluation.case_persistence import (
    build_frozen_jsonl,
    frozen_content_sha256,
    write_case_jsonl,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    DatasetStatus,
    Difficulty,
    EvalCaseType,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
    SourceDocumentType,
)


def _case() -> ReviewedEvalCase:
    return ReviewedEvalCase(
        case_code="CASE_EXACT_001",
        question="收到错误码F-ORDER-001时应该怎么处理？",
        reference_answer="已发货订单不能直接取消，应走售后流程。",
        case_type=EvalCaseType.EXACT,
        target_doc_types=[SourceDocumentType.FAQ],
        expected_action=ExpectedAction.ANSWER,
        difficulty=Difficulty.EASY,
        dataset_split=DatasetSplit.DEVELOPMENT,
        business_domain="ecommerce_after_sales",
        required_fact_count=1,
        generation_metadata={
            "required_identifier": "F-ORDER-001",
            "source_doc_codes": ["FAQ_ORDER_STATUS_001"],
        },
        evidences=[
            EvidenceSpec(
                source_doc_code="FAQ_ORDER_STATUS_001",
                evidence_quote="已发货订单不能直接取消，应走售后流程",
                fact_key="order_cancel_shipped",
                relevance_grade=3,
                mapped_doc_id=15,
                mapped_chunk_id=54,
                mapping_status=MappingStatus.MAPPED,
            )
        ],
        review_status=ReviewStatus.PASSED,
        review_score=0.99,
    )


SOURCE_ROW = {
    "id": 10,
    "source_doc_code": "FAQ_ORDER_STATUS_001",
    "title": "订单状态FAQ",
    "doc_type": "FAQ",
    "topic": "order",
    "version": "1.0",
    "effective_from": None,
    "effective_to": None,
    "is_current": 1,
    "relative_file_path": "evaluation/corpus/source/faq.json",
    "source_content_sha256": "a" * 64,
    "generation_spec_json": {},
    "review_status": "PASSED",
    "review_score": 0.99,
    "review_reason": "通过",
    "mapped_doc_id": 15,
}


class FakeRepository:
    def __init__(self, target=None, source_digest=None) -> None:
        self.target = target
        self.source_digest = source_digest
        self.saved_documents = []
        self.saved_cases = []
        self.saved_evidences = []
        self.frozen = None

    def find_dataset(self, code, version):
        if version == "v1":
            return {
                "id": 1,
                "dataset_code": code,
                "version": "v1",
                "status": "FROZEN",
                "content_sha256": self.source_digest,
            }
        return self.target

    def create_dataset(self, **kwargs):
        self.created_dataset = kwargs
        return 2

    def list_source_documents(self, dataset_id):
        if dataset_id == 1:
            return [SOURCE_ROW]
        return [
            {
                **SOURCE_ROW,
                "id": 20,
                "dataset_id": 2,
            }
        ]

    def upsert_source_document(self, dataset_id, document):
        self.saved_documents.append((dataset_id, document))
        return 20

    def upsert_eval_case(self, dataset_id, case):
        self.saved_cases.append((dataset_id, case))
        return 200

    def delete_case_evidence(self, case_id):
        self.deleted_case_id = case_id

    def save_case_evidence(
        self,
        case_id,
        source_document_id,
        evidence,
    ):
        self.saved_evidences.append(
            (case_id, source_document_id, evidence)
        )
        return len(self.saved_evidences)

    def freeze_dataset(self, dataset_id, digest):
        self.frozen = (dataset_id, digest)


class FakeDocumentRepository:
    def list_chunks_by_doc_id(self, doc_id):
        assert doc_id == 15
        return [
            {
                "id": 54,
                "doc_id": 15,
                "sort_order": 5,
                "title": "已发货订单如何取消？",
                "content": "已发货订单不能直接取消，应走售后流程。",
            },
            {
                "id": 57,
                "doc_id": 15,
                "sort_order": 8,
                "title": "提示F-ORDER-001是什么意思？",
                "content": "错误码F-ORDER-001表示当前订单状态不允许取消。",
            },
        ]


class FakeValidator:
    def validate(self, source_documents, cases):
        return SimpleNamespace(
            is_valid=True,
            document_count=len(source_documents),
            case_count=len(cases),
            issues=[],
        )


def _args(tmp_path):
    input_path = tmp_path / "v1.jsonl"
    cases = [_case()]
    write_case_jsonl(input_path, cases)
    return SimpleNamespace(
        input=input_path,
        output=tmp_path / "v2.frozen.jsonl",
        report=tmp_path / "v2.correction.json",
        dataset_code="rag_eval_ecommerce",
        source_version="v1",
        target_version="v2",
        name="修正版RAG评测集",
    )


def _source_digest() -> str:
    return frozen_content_sha256(build_frozen_jsonl([_case()]))


def test_creates_corrected_frozen_dataset_and_audit_report(
    tmp_path,
) -> None:
    repository = FakeRepository(source_digest=_source_digest())
    args = _args(tmp_path)

    result = create_corrected_dataset(
        args=args,
        repository=repository,
        document_repository=FakeDocumentRepository(),
        validator=FakeValidator(),
    )

    assert result["dataset"] == "rag_eval_ecommerce:v2"
    assert result["corrected_case_count"] == 1
    assert len(repository.saved_documents) == 1
    assert len(repository.saved_cases) == 1
    assert len(repository.saved_evidences) == 2
    assert repository.frozen[0] == 2
    assert args.output.exists()
    assert args.report.exists()


def test_refuses_to_overwrite_frozen_v2_with_different_digest(
    tmp_path,
) -> None:
    repository = FakeRepository(
        target={
            "id": 2,
            "status": DatasetStatus.FROZEN.value,
            "content_sha256": "0" * 64,
        },
        source_digest=_source_digest(),
    )

    with pytest.raises(ValueError, match="摘要不一致"):
        create_corrected_dataset(
            args=_args(tmp_path),
            repository=repository,
            document_repository=FakeDocumentRepository(),
            validator=FakeValidator(),
        )


def test_refuses_input_that_does_not_match_frozen_source(
    tmp_path,
) -> None:
    repository = FakeRepository(source_digest="b" * 64)

    with pytest.raises(ValueError, match="输入文件摘要"):
        create_corrected_dataset(
            args=_args(tmp_path),
            repository=repository,
            document_repository=FakeDocumentRepository(),
            validator=FakeValidator(),
        )
