from src.rag_platform.evaluation.dataset_validator import (
    DatasetValidationPolicy,
    DatasetValidator,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseStatus,
    EvalCaseType,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
    SourceDocumentSpec,
    SourceDocumentType,
)


def make_document(
    *,
    source_doc_code: str = "RULE_REFUND_001",
    mapped_doc_id: int | None = 10,
    review_status: ReviewStatus = ReviewStatus.PASSED,
    review_score: float = 0.95,
) -> SourceDocumentSpec:
    return SourceDocumentSpec(
        source_doc_code=source_doc_code,
        title="退款规则",
        doc_type=SourceDocumentType.RULE,
        topic="refund",
        relative_file_path="rules/refund.docx",
        source_content_sha256="a" * 64,
        review_status=review_status,
        review_score=review_score,
        mapped_doc_id=mapped_doc_id,
    )


def make_case(
    *,
    case_code: str = "CASE_001",
    source_doc_code: str = "RULE_REFUND_001",
    split: DatasetSplit = DatasetSplit.DEVELOPMENT,
    mapping_status: MappingStatus = MappingStatus.MAPPED,
    mapped_chunk_id: int | None = 101,
    source_group: str = "refund-v1",
) -> ReviewedEvalCase:
    evidence_kwargs = {
        "source_doc_code": source_doc_code,
        "evidence_quote": "未发货订单允许直接申请退款。",
        "fact_key": "refund_rule",
        "relevance_grade": 3,
        "mapping_status": mapping_status,
    }
    if mapping_status == MappingStatus.MAPPED:
        evidence_kwargs.update(
            mapped_doc_id=10,
            mapped_chunk_id=mapped_chunk_id,
        )

    return ReviewedEvalCase(
        case_code=case_code,
        question=f"{case_code}：未发货订单可以退款吗？",
        normalized_question=f"{case_code.lower()} 未发货订单可以退款吗",
        reference_answer="未发货订单允许直接申请退款。",
        case_type=EvalCaseType.DIRECT,
        target_doc_types=[SourceDocumentType.RULE],
        expected_action=ExpectedAction.ANSWER,
        dataset_split=split,
        required_fact_count=1,
        generation_metadata={"source_group": source_group},
        evidences=[EvidenceSpec(**evidence_kwargs)],
        review_status=ReviewStatus.PASSED,
        review_score=0.95,
        status=EvalCaseStatus.ACTIVE,
    )


def make_policy(
    *,
    case_count: int = 1,
    split_counts: dict[DatasetSplit, int] | None = None,
) -> DatasetValidationPolicy:
    return DatasetValidationPolicy(
        document_type_counts={SourceDocumentType.RULE: 1},
        case_type_counts={EvalCaseType.DIRECT: case_count},
        split_counts=split_counts or {DatasetSplit.DEVELOPMENT: case_count},
        min_document_review_score=0.8,
        min_case_review_score=0.8,
    )


def test_validator_accepts_complete_mapped_dataset() -> None:
    report = DatasetValidator(make_policy()).validate(
        documents=[make_document()],
        cases=[make_case()],
    )

    assert report.is_valid is True
    assert report.issues == []


def test_default_policy_requires_document_review_score_of_at_least_088() -> None:
    policy = DatasetValidationPolicy(
        document_type_counts={SourceDocumentType.RULE: 1},
        case_type_counts={EvalCaseType.DIRECT: 1},
        split_counts={DatasetSplit.DEVELOPMENT: 1},
    )

    report = DatasetValidator(policy).validate(
        documents=[make_document(review_score=0.87)],
        cases=[make_case()],
    )

    assert "DOCUMENT_REVIEW_SCORE_TOO_LOW" in {
        issue.code for issue in report.issues
    }


def test_validator_rejects_unreviewed_or_unmapped_document() -> None:
    report = DatasetValidator(make_policy()).validate(
        documents=[
            make_document(
                mapped_doc_id=None,
                review_status=ReviewStatus.PENDING,
            )
        ],
        cases=[make_case()],
    )

    assert report.is_valid is False
    assert {issue.code for issue in report.issues} >= {
        "DOCUMENT_NOT_REVIEWED",
        "DOCUMENT_NOT_MAPPED",
    }


def test_validator_rejects_missing_evidence_mapping() -> None:
    report = DatasetValidator(make_policy()).validate(
        documents=[make_document()],
        cases=[
            make_case(
                mapping_status=MappingStatus.MISSING,
                mapped_chunk_id=None,
            )
        ],
    )

    assert report.is_valid is False
    assert "EVIDENCE_NOT_MAPPED" in {issue.code for issue in report.issues}


def test_validator_rejects_duplicate_codes_and_quota_mismatch() -> None:
    report = DatasetValidator(
        make_policy(
            case_count=2,
            split_counts={DatasetSplit.DEVELOPMENT: 2},
        )
    ).validate(
        documents=[make_document(), make_document()],
        cases=[make_case(), make_case()],
    )

    assert report.is_valid is False
    assert {issue.code for issue in report.issues} >= {
        "DUPLICATE_DOCUMENT_CODE",
        "DUPLICATE_CASE_CODE",
        "DOCUMENT_TYPE_COUNT_MISMATCH",
    }


def test_validator_rejects_source_group_split_leakage() -> None:
    report = DatasetValidator(
        make_policy(
            case_count=2,
            split_counts={
                DatasetSplit.DEVELOPMENT: 1,
                DatasetSplit.TEST: 1,
            },
        )
    ).validate(
        documents=[make_document()],
        cases=[
            make_case(
                case_code="CASE_DEV",
                split=DatasetSplit.DEVELOPMENT,
                source_group="refund-rule-family",
            ),
            make_case(
                case_code="CASE_TEST",
                split=DatasetSplit.TEST,
                source_group="refund-rule-family",
            ),
        ],
    )

    assert report.is_valid is False
    assert "SOURCE_GROUP_SPLIT_LEAKAGE" in {
        issue.code for issue in report.issues
    }
