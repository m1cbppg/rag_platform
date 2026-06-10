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
from src.rag_platform.evaluation.models import (  # noqa: E402
    ReviewStatus,
    SourceDocumentSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验、导出并冻结M4的300道评测题",
    )
    parser.add_argument("--dataset-id", type=int, default=7)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.mapped.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.frozen.jsonl"),
    )
    return parser.parse_args()


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
    cases = load_reviewed_case_jsonl(args.input)
    mapping_errors = validate_required_evidence_mapped(cases)
    if mapping_errors:
        raise ValueError("；".join(mapping_errors))

    repository = DatasetRepository()
    source_documents = [
        _source_document(row)
        for row in repository.list_source_documents(args.dataset_id)
    ]
    report = DatasetValidator().validate(source_documents, cases)
    if not report.is_valid:
        raise ValueError(
            "数据集冻结校验失败："
            + "；".join(
                f"{issue.code}:{issue.message}"
                for issue in report.issues
            )
        )

    content = build_frozen_jsonl(cases)
    digest = frozen_content_sha256(content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    repository.freeze_dataset(args.dataset_id, digest)
    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "document_count": report.document_count,
                "case_count": report.case_count,
                "content_sha256": digest,
                "status": "FROZEN",
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
