import argparse
import json
from pathlib import Path
import sys

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.action_calibration import (  # noqa: E402
    validate_action_calibration_cases,
)
from src.rag_platform.evaluation.case_persistence import (  # noqa: E402
    build_frozen_jsonl,
    frozen_content_sha256,
    load_reviewed_case_jsonl,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.models import DatasetStatus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="注册并冻结RAG动作决策校准集",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "evaluation/datasets/rag_action_calibration_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/datasets/"
            "rag_action_calibration_v1.frozen.jsonl"
        ),
    )
    parser.add_argument("--dataset-code", default="rag_eval_action")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--name", default="RAG动作决策校准集")
    return parser.parse_args()


def register_action_calibration(
    *,
    args: argparse.Namespace,
    repository: DatasetRepository,
    active_chunk_ids: set[int],
) -> dict:
    cases = load_reviewed_case_jsonl(args.input)
    errors = validate_action_calibration_cases(
        cases,
        active_chunk_ids=active_chunk_ids,
    )
    if errors:
        raise ValueError("；".join(errors))

    content = build_frozen_jsonl(cases)
    digest = frozen_content_sha256(content)
    existing = repository.find_dataset(
        args.dataset_code,
        args.version,
    )
    if existing is not None:
        dataset_id = int(existing["id"])
        if existing["status"] == DatasetStatus.FROZEN.value:
            if existing.get("content_sha256") != digest:
                raise ValueError(
                    "同版本动作校准集已冻结，内容摘要不一致"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
            return {
                "dataset_id": dataset_id,
                "dataset": f"{args.dataset_code}:{args.version}",
                "case_count": len(cases),
                "content_sha256": digest,
                "status": DatasetStatus.FROZEN.value,
                "reused": True,
                "output": args.output.as_posix(),
            }
    else:
        dataset_id = repository.create_dataset(
            dataset_code=args.dataset_code,
            name=args.name,
            version=args.version,
            domain="ecommerce_after_sales",
            description=(
                "用于校准ANSWER、REFUSE、CLARIFY动作决策的独立评测集；"
                "CLARIFY题必须包含缺失条件、标准追问和来源Chunk分支。"
            ),
            generator_provider="human",
            generator_model=None,
            reviewer_provider="human",
            reviewer_model=None,
            generation_config={
                "dataset_type": "action_calibration",
                "clarification_contract_version": "v1",
                "input_path": args.input.as_posix(),
            },
            status=DatasetStatus.REVIEWED,
        )

    for case in cases:
        repository.upsert_eval_case(dataset_id, case)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    repository.freeze_dataset(dataset_id, digest)
    return {
        "dataset_id": dataset_id,
        "dataset": f"{args.dataset_code}:{args.version}",
        "case_count": len(cases),
        "content_sha256": digest,
        "status": DatasetStatus.FROZEN.value,
        "reused": False,
        "output": args.output.as_posix(),
    }


def run(args: argparse.Namespace) -> int:
    repository = DatasetRepository()
    with repository.engine.begin() as connection:
        active_chunk_ids = {
            int(row["id"])
            for row in connection.execute(
                text(
                    """
                    SELECT id
                    FROM rag_chunk
                    WHERE status = 'ACTIVE'
                    """
                )
            ).mappings()
        }
    result = register_action_calibration(
        args=args,
        repository=repository,
        active_chunk_ids=active_chunk_ids,
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
