import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.case_persistence import (  # noqa: E402
    load_reviewed_case_jsonl,
    validate_required_evidence_mapped,
    write_case_jsonl,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.evidence_mapper import (  # noqa: E402
    map_case_evidence,
)
from src.rag_platform.evaluation.models import DatasetStatus  # noqa: E402
from src.rag_platform.infrastructure.repositories.document_repository import (  # noqa: E402
    DocumentRepository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把M4标准证据确定性映射到rag_chunk并写入MySQL",
    )
    parser.add_argument("--dataset-id", type=int, default=7)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.reviewed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.mapped.jsonl"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    repository = DatasetRepository()
    dataset = repository.find_dataset("rag_eval_ecommerce", "v1")
    if dataset is None or int(dataset["id"]) != args.dataset_id:
        raise ValueError("目标评测数据集不存在或dataset_id不匹配")
    if dataset["status"] == DatasetStatus.FROZEN.value:
        raise ValueError("数据集已经冻结，禁止原地修改")

    source_rows = repository.list_source_documents(args.dataset_id)
    source_by_code = {
        row["source_doc_code"]: row for row in source_rows
    }
    document_repository = DocumentRepository()
    chunks_by_code = {
        code: document_repository.list_chunks_by_doc_id(
            int(row["mapped_doc_id"])
        )
        for code, row in source_by_code.items()
        if row.get("mapped_doc_id") is not None
    }

    cases = load_reviewed_case_jsonl(args.input)
    mapped_cases = [
        map_case_evidence(
            case=case,
            source_documents=source_by_code,
            chunks_by_source_code=chunks_by_code,
        )
        for case in cases
    ]
    write_case_jsonl(args.output, mapped_cases)
    errors = validate_required_evidence_mapped(mapped_cases)
    if errors:
        print(
            json.dumps(
                {"mapping_errors": errors},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    for case in mapped_cases:
        case_id = repository.upsert_eval_case(args.dataset_id, case)
        repository.delete_case_evidence(case_id)
        for evidence in case.evidences:
            source_document_id = int(
                source_by_code[evidence.source_doc_code]["id"]
            )
            repository.save_case_evidence(
                case_id=case_id,
                source_document_id=source_document_id,
                evidence=evidence,
            )
    repository.update_dataset_status(
        args.dataset_id,
        DatasetStatus.REVIEWED,
    )
    print(
        json.dumps(
            {
                "mapped_cases": len(mapped_cases),
                "mapped_evidences": sum(
                    len(case.evidences) for case in mapped_cases
                ),
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
