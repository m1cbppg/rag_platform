import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.case_persistence import (  # noqa: E402
    build_frozen_jsonl,
    frozen_content_sha256,
    load_reviewed_case_jsonl,
    validate_required_evidence_mapped,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.dataset_validator import (  # noqa: E402
    DatasetValidator,
)
from src.rag_platform.evaluation.exact_evidence_correction import (  # noqa: E402
    EXACT_EVIDENCE_CORRECTION_VERSION,
    correct_exact_identifier_evidence,
)
from src.rag_platform.evaluation.models import (  # noqa: E402
    DatasetStatus,
    ReviewStatus,
    SourceDocumentSpec,
)
from src.rag_platform.infrastructure.repositories.document_repository import (  # noqa: E402
    DocumentRepository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于冻结v1创建修正EXACT identifier证据的v2评测集",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "evaluation/datasets/rag_eval_v1.frozen.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/datasets/rag_eval_v2.frozen.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "evaluation/reports/rag_eval_v2_correction.json"
        ),
    )
    parser.add_argument(
        "--dataset-code",
        default="rag_eval_ecommerce",
    )
    parser.add_argument("--source-version", default="v1")
    parser.add_argument("--target-version", default="v2")
    parser.add_argument(
        "--name",
        default="电商售后RAG评测集（EXACT证据修正版）",
    )
    return parser.parse_args()


def create_corrected_dataset(
    *,
    args: argparse.Namespace,
    repository: DatasetRepository,
    document_repository: DocumentRepository,
    validator: DatasetValidator,
) -> dict:
    source_dataset = repository.find_dataset(
        args.dataset_code,
        args.source_version,
    )
    if source_dataset is None:
        raise ValueError("源评测数据集不存在")
    if source_dataset["status"] != DatasetStatus.FROZEN.value:
        raise ValueError("源评测数据集必须处于FROZEN状态")

    source_cases = load_reviewed_case_jsonl(args.input)
    source_digest = frozen_content_sha256(
        build_frozen_jsonl(source_cases)
    )
    if source_digest != source_dataset.get("content_sha256"):
        raise ValueError(
            "输入文件摘要与数据库中冻结的源评测集不一致"
        )

    source_rows = repository.list_source_documents(
        int(source_dataset["id"])
    )
    source_documents = [
        _source_document(row)
        for row in source_rows
    ]
    chunks_by_doc_id = {
        int(row["mapped_doc_id"]): (
            document_repository.list_chunks_by_doc_id(
                int(row["mapped_doc_id"])
            )
        )
        for row in source_rows
        if row.get("mapped_doc_id") is not None
    }
    corrected_cases, correction_report = (
        correct_exact_identifier_evidence(
            cases=source_cases,
            source_documents=source_rows,
            chunks_by_doc_id=chunks_by_doc_id,
        )
    )
    mapping_errors = validate_required_evidence_mapped(
        corrected_cases
    )
    if mapping_errors:
        raise ValueError("；".join(mapping_errors))

    validation_report = validator.validate(
        source_documents,
        corrected_cases,
    )
    if not validation_report.is_valid:
        raise ValueError(
            "修正版数据集校验失败："
            + "；".join(
                f"{issue.code}:{issue.message}"
                for issue in validation_report.issues
            )
        )

    content = build_frozen_jsonl(corrected_cases)
    digest = frozen_content_sha256(content)
    target_dataset = repository.find_dataset(
        args.dataset_code,
        args.target_version,
    )
    if (
        target_dataset is not None
        and target_dataset["status"]
        == DatasetStatus.FROZEN.value
    ):
        if target_dataset.get("content_sha256") != digest:
            raise ValueError(
                "同版本修正版数据集已冻结，内容摘要不一致"
            )
        result = _write_artifacts(
            args=args,
            content=content,
            digest=digest,
            source_dataset=source_dataset,
            target_dataset_id=int(target_dataset["id"]),
            correction_report=correction_report,
            document_count=validation_report.document_count,
            case_count=validation_report.case_count,
            reused=True,
        )
        return result

    if target_dataset is None:
        dataset_id = repository.create_dataset(
            dataset_code=args.dataset_code,
            name=args.name,
            version=args.target_version,
            domain="ecommerce_after_sales",
            description=(
                "从冻结v1复制，补充EXACT题required_identifier所在"
                "Chunk作为独立必要证据；问题、答案和数据划分保持不变。"
            ),
            generator_provider="deterministic",
            generator_model=None,
            reviewer_provider="human+qwen",
            reviewer_model=None,
            generation_config={
                "dataset_type": "corrected_evaluation",
                "source_version": args.source_version,
                "source_content_sha256": source_dataset.get(
                    "content_sha256"
                ),
                "correction_version": (
                    EXACT_EVIDENCE_CORRECTION_VERSION
                ),
            },
            status=DatasetStatus.REVIEWED,
        )
    else:
        dataset_id = int(target_dataset["id"])

    for document in source_documents:
        repository.upsert_source_document(
            dataset_id,
            document,
        )
    target_source_rows = repository.list_source_documents(dataset_id)
    target_source_id_by_code = {
        str(row["source_doc_code"]): int(row["id"])
        for row in target_source_rows
    }
    for case in corrected_cases:
        case_id = repository.upsert_eval_case(dataset_id, case)
        repository.delete_case_evidence(case_id)
        for evidence in case.evidences:
            source_document_id = target_source_id_by_code.get(
                evidence.source_doc_code
            )
            if source_document_id is None:
                raise ValueError(
                    "修正版数据集缺少源文档："
                    f"{evidence.source_doc_code}"
                )
            repository.save_case_evidence(
                case_id=case_id,
                source_document_id=source_document_id,
                evidence=evidence,
            )

    repository.freeze_dataset(dataset_id, digest)
    return _write_artifacts(
        args=args,
        content=content,
        digest=digest,
        source_dataset=source_dataset,
        target_dataset_id=dataset_id,
        correction_report=correction_report,
        document_count=validation_report.document_count,
        case_count=validation_report.case_count,
        reused=False,
    )


def _write_artifacts(
    *,
    args: argparse.Namespace,
    content: str,
    digest: str,
    source_dataset: dict,
    target_dataset_id: int,
    correction_report: dict,
    document_count: int,
    case_count: int,
    reused: bool,
) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    report = {
        **correction_report,
        "source_dataset": (
            f"{args.dataset_code}:{args.source_version}"
        ),
        "source_dataset_id": int(source_dataset["id"]),
        "source_content_sha256": source_dataset.get(
            "content_sha256"
        ),
        "target_dataset": (
            f"{args.dataset_code}:{args.target_version}"
        ),
        "target_dataset_id": target_dataset_id,
        "target_content_sha256": digest,
        "document_count": document_count,
        "case_count": case_count,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "dataset_id": target_dataset_id,
        "dataset": (
            f"{args.dataset_code}:{args.target_version}"
        ),
        "document_count": document_count,
        "case_count": case_count,
        "corrected_case_count": correction_report[
            "corrected_case_count"
        ],
        "content_sha256": digest,
        "status": DatasetStatus.FROZEN.value,
        "reused": reused,
        "output": args.output.as_posix(),
        "report": args.report.as_posix(),
    }


def _source_document(row: dict) -> SourceDocumentSpec:
    return SourceDocumentSpec(
        source_doc_code=row["source_doc_code"],
        title=row["title"],
        doc_type=row["doc_type"],
        topic=row["topic"],
        version=row["version"],
        effective_from=row["effective_from"],
        effective_to=row["effective_to"],
        is_current=bool(row["is_current"]),
        relative_file_path=row["relative_file_path"],
        source_content_sha256=row["source_content_sha256"],
        generation_spec=row["generation_spec_json"],
        review_status=ReviewStatus(row["review_status"]),
        review_score=(
            float(row["review_score"])
            if row["review_score"] is not None
            else None
        ),
        review_reason=row["review_reason"],
        mapped_doc_id=row["mapped_doc_id"],
    )


def run(args: argparse.Namespace) -> int:
    result = create_corrected_dataset(
        args=args,
        repository=DatasetRepository(),
        document_repository=DocumentRepository(),
        validator=DatasetValidator(),
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
