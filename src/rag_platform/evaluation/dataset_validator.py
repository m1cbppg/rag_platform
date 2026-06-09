from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseStatus,
    EvalCaseType,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
    SourceDocumentSpec,
    SourceDocumentType,
)


def default_document_type_counts() -> dict[SourceDocumentType, int]:
    return {
        SourceDocumentType.FAQ: 12,
        SourceDocumentType.SOP: 10,
        SourceDocumentType.RULE: 12,
        SourceDocumentType.MANUAL: 6,
    }


def default_case_type_counts() -> dict[EvalCaseType, int]:
    return {
        EvalCaseType.DIRECT: 90,
        EvalCaseType.PARAPHRASE: 45,
        EvalCaseType.EXACT: 30,
        EvalCaseType.MULTI_CONDITION: 45,
        EvalCaseType.MULTI_HOP: 30,
        EvalCaseType.CONFLICT: 20,
        EvalCaseType.NO_ANSWER: 40,
    }


def default_split_counts() -> dict[DatasetSplit, int]:
    return {
        DatasetSplit.DEVELOPMENT: 180,
        DatasetSplit.VALIDATION: 60,
        DatasetSplit.TEST: 60,
    }


@dataclass(frozen=True)
class DatasetValidationPolicy:
    document_type_counts: dict[SourceDocumentType, int] = field(
        default_factory=default_document_type_counts
    )
    case_type_counts: dict[EvalCaseType, int] = field(
        default_factory=default_case_type_counts
    )
    split_counts: dict[DatasetSplit, int] = field(
        default_factory=default_split_counts
    )
    min_document_review_score: float = 0.88
    min_case_review_score: float | None = None
    require_mapped_documents: bool = True
    require_mapped_evidence: bool = True


@dataclass(frozen=True)
class DatasetValidationIssue:
    code: str
    message: str
    entity_code: str | None = None


@dataclass(frozen=True)
class DatasetValidationReport:
    issues: list[DatasetValidationIssue]
    document_count: int
    case_count: int

    @property
    def is_valid(self) -> bool:
        return not self.issues


class DatasetValidator:
    def __init__(self, policy: DatasetValidationPolicy | None = None) -> None:
        self.policy = policy or DatasetValidationPolicy()

    def validate(
        self,
        documents: list[SourceDocumentSpec],
        cases: list[ReviewedEvalCase],
    ) -> DatasetValidationReport:
        issues: list[DatasetValidationIssue] = []

        self._validate_documents(documents, issues)
        self._validate_cases(documents, cases, issues)
        self._validate_distribution(documents, cases, issues)
        self._validate_split_isolation(cases, issues)

        return DatasetValidationReport(
            issues=issues,
            document_count=len(documents),
            case_count=len(cases),
        )

    def _validate_documents(
        self,
        documents: list[SourceDocumentSpec],
        issues: list[DatasetValidationIssue],
    ) -> None:
        seen_codes: set[str] = set()

        for document in documents:
            if document.source_doc_code in seen_codes:
                issues.append(
                    DatasetValidationIssue(
                        code="DUPLICATE_DOCUMENT_CODE",
                        message="源文档编码重复",
                        entity_code=document.source_doc_code,
                    )
                )
            seen_codes.add(document.source_doc_code)

            if document.review_status != ReviewStatus.PASSED:
                issues.append(
                    DatasetValidationIssue(
                        code="DOCUMENT_NOT_REVIEWED",
                        message="源文档尚未通过独立审核",
                        entity_code=document.source_doc_code,
                    )
                )

            if (
                document.review_score is None
                or document.review_score
                < self.policy.min_document_review_score
            ):
                issues.append(
                    DatasetValidationIssue(
                        code="DOCUMENT_REVIEW_SCORE_TOO_LOW",
                        message=(
                            "源文档审核分数低于冻结阈值 "
                            f"{self.policy.min_document_review_score}"
                        ),
                        entity_code=document.source_doc_code,
                    )
                )

            if (
                self.policy.require_mapped_documents
                and document.mapped_doc_id is None
            ):
                issues.append(
                    DatasetValidationIssue(
                        code="DOCUMENT_NOT_MAPPED",
                        message="源文档尚未映射到 rag_document",
                        entity_code=document.source_doc_code,
                    )
                )

    def _validate_cases(
        self,
        documents: list[SourceDocumentSpec],
        cases: list[ReviewedEvalCase],
        issues: list[DatasetValidationIssue],
    ) -> None:
        documents_by_code = {
            document.source_doc_code: document
            for document in documents
        }
        seen_case_codes: set[str] = set()
        seen_questions: set[str] = set()

        for case in cases:
            if case.case_code in seen_case_codes:
                issues.append(
                    DatasetValidationIssue(
                        code="DUPLICATE_CASE_CODE",
                        message="评测题编码重复",
                        entity_code=case.case_code,
                    )
                )
            seen_case_codes.add(case.case_code)

            normalized_question = (
                case.normalized_question or case.question
            ).strip().casefold()
            if normalized_question in seen_questions:
                issues.append(
                    DatasetValidationIssue(
                        code="DUPLICATE_QUESTION",
                        message="存在重复或规范化后重复的问题",
                        entity_code=case.case_code,
                    )
                )
            seen_questions.add(normalized_question)

            if case.review_status != ReviewStatus.PASSED:
                issues.append(
                    DatasetValidationIssue(
                        code="CASE_NOT_REVIEWED",
                        message="评测题尚未通过独立审核",
                        entity_code=case.case_code,
                    )
                )

            if (
                self.policy.min_case_review_score is not None
                and (
                    case.review_score is None
                    or case.review_score < self.policy.min_case_review_score
                )
            ):
                issues.append(
                    DatasetValidationIssue(
                        code="CASE_REVIEW_SCORE_TOO_LOW",
                        message=(
                            "评测题审核分数低于冻结阈值 "
                            f"{self.policy.min_case_review_score}"
                        ),
                        entity_code=case.case_code,
                    )
                )

            if case.status != EvalCaseStatus.ACTIVE:
                issues.append(
                    DatasetValidationIssue(
                        code="CASE_NOT_ACTIVE",
                        message="评测题不是 ACTIVE 状态",
                        entity_code=case.case_code,
                    )
                )

            if case.dataset_split == DatasetSplit.UNASSIGNED:
                issues.append(
                    DatasetValidationIssue(
                        code="CASE_SPLIT_UNASSIGNED",
                        message="评测题尚未分配数据集划分",
                        entity_code=case.case_code,
                    )
                )

            for evidence in case.evidences:
                source_document = documents_by_code.get(evidence.source_doc_code)

                if source_document is None:
                    issues.append(
                        DatasetValidationIssue(
                            code="EVIDENCE_SOURCE_NOT_FOUND",
                            message="标准证据引用的源文档不存在",
                            entity_code=case.case_code,
                        )
                    )
                    continue

                if self.policy.require_mapped_evidence:
                    if (
                        evidence.mapping_status != MappingStatus.MAPPED
                        or evidence.mapped_doc_id is None
                        or evidence.mapped_chunk_id is None
                    ):
                        issues.append(
                            DatasetValidationIssue(
                                code="EVIDENCE_NOT_MAPPED",
                                message="标准证据尚未无歧义映射到 Chunk",
                                entity_code=case.case_code,
                            )
                        )
                    elif evidence.mapped_doc_id != source_document.mapped_doc_id:
                        issues.append(
                            DatasetValidationIssue(
                                code="EVIDENCE_DOCUMENT_MAPPING_MISMATCH",
                                message="证据映射文档与源文档映射结果不一致",
                                entity_code=case.case_code,
                            )
                        )

    def _validate_distribution(
        self,
        documents: list[SourceDocumentSpec],
        cases: list[ReviewedEvalCase],
        issues: list[DatasetValidationIssue],
    ) -> None:
        actual_document_counts = Counter(
            document.doc_type for document in documents
        )
        actual_case_counts = Counter(case.case_type for case in cases)
        actual_split_counts = Counter(case.dataset_split for case in cases)

        self._append_count_issues(
            actual=actual_document_counts,
            expected=self.policy.document_type_counts,
            issue_code="DOCUMENT_TYPE_COUNT_MISMATCH",
            label="文档类型",
            issues=issues,
        )
        self._append_count_issues(
            actual=actual_case_counts,
            expected=self.policy.case_type_counts,
            issue_code="CASE_TYPE_COUNT_MISMATCH",
            label="评测题类型",
            issues=issues,
        )
        self._append_count_issues(
            actual=actual_split_counts,
            expected=self.policy.split_counts,
            issue_code="SPLIT_COUNT_MISMATCH",
            label="数据集划分",
            issues=issues,
        )

    def _append_count_issues(
        self,
        actual: Counter,
        expected: dict,
        issue_code: str,
        label: str,
        issues: list[DatasetValidationIssue],
    ) -> None:
        all_keys = set(actual) | set(expected)

        for key in sorted(all_keys, key=str):
            actual_count = actual.get(key, 0)
            expected_count = expected.get(key, 0)

            if actual_count != expected_count:
                issues.append(
                    DatasetValidationIssue(
                        code=issue_code,
                        message=(
                            f"{label} {key} 数量应为 {expected_count}，"
                            f"实际为 {actual_count}"
                        ),
                    )
                )

    def _validate_split_isolation(
        self,
        cases: list[ReviewedEvalCase],
        issues: list[DatasetValidationIssue],
    ) -> None:
        group_splits: dict[str, set[DatasetSplit]] = defaultdict(set)

        for case in cases:
            source_group = case.generation_metadata.get("source_group")
            if source_group:
                group_splits[str(source_group)].add(case.dataset_split)

        for source_group, splits in group_splits.items():
            if len(splits) > 1:
                issues.append(
                    DatasetValidationIssue(
                        code="SOURCE_GROUP_SPLIT_LEAKAGE",
                        message=(
                            f"来源组 {source_group} 同时出现在多个数据划分："
                            f"{sorted(str(item) for item in splits)}"
                        ),
                        entity_code=source_group,
                    )
                )
