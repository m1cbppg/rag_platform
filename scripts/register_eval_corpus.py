import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.core.config import get_settings  # noqa: E402
from src.rag_platform.evaluation.corpus_models import (  # noqa: E402
    ReviewedDocumentOutcome,
)
from src.rag_platform.evaluation.corpus_persistence import (  # noqa: E402
    build_source_document_spec,
)
from src.rag_platform.evaluation.corpus_validation import (  # noqa: E402
    load_document_blueprints,
    validate_blueprint_plan,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.models import DatasetStatus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="注册M2评测语料到MySQL")
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path("evaluation/blueprints/ecommerce_document_plan.json"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("evaluation/corpus/source"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("evaluation/corpus/reviews"),
    )
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--dataset-code", default="rag_eval_ecommerce")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--name", default="电商售后RAG评测集")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    blueprints = load_document_blueprints(args.blueprint)
    report = validate_blueprint_plan(blueprints)
    if not report.is_valid:
        raise ValueError("蓝图校验失败：" + "；".join(report.errors))

    settings = get_settings()
    repository = DatasetRepository()
    dataset_id = args.dataset_id
    if dataset_id is None:
        existing = repository.find_dataset(
            args.dataset_code,
            args.version,
        )
        if existing is not None:
            dataset_id = int(existing["id"])
        else:
            blueprint_sha256 = hashlib.sha256(
                args.blueprint.read_bytes()
            ).hexdigest()
            dataset_id = repository.create_dataset(
                dataset_code=args.dataset_code,
                name=args.name,
                version=args.version,
                domain="ecommerce_after_sales",
                description=(
                    "包含40篇受控电商售后源文档的RAG评测数据集；"
                    "DeepSeek生成，百炼Qwen独立审核。"
                ),
                generator_provider="deepseek",
                generator_model=settings.deepseek_chat_model,
                reviewer_provider="dashscope",
                reviewer_model=settings.qwen_judge_model,
                generation_config={
                    "blueprint_path": args.blueprint.as_posix(),
                    "blueprint_sha256": blueprint_sha256,
                    "document_count": 40,
                    "document_distribution": {
                        "FAQ": 12,
                        "SOP": 10,
                        "RULE": 12,
                        "MANUAL": 6,
                    },
                    "generator_temperature": 0.4,
                    "max_generation_attempts": 3,
                    "max_regeneration_rounds": 2,
                },
            )

    registered = 0
    for blueprint in blueprints:
        source_path = args.source / f"{blueprint.source_doc_code}.json"
        review_path = args.reviews / f"{blueprint.source_doc_code}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"缺少源文档：{source_path}")
        if not review_path.exists():
            raise FileNotFoundError(f"缺少审核结果：{review_path}")

        outcome = ReviewedDocumentOutcome.model_validate_json(
            review_path.read_text(encoding="utf-8")
        )
        if not outcome.review.passed:
            raise ValueError(
                f"{blueprint.source_doc_code} 未通过审核，禁止注册"
            )
        spec = build_source_document_spec(
            blueprint=blueprint,
            document=outcome.document,
            source_path=source_path,
            outcome=outcome,
            project_root=PROJECT_ROOT,
        )
        repository.upsert_source_document(dataset_id, spec)
        registered += 1

    repository.update_dataset_status(
        dataset_id,
        DatasetStatus.REVIEWED,
    )
    dataset = repository.find_dataset(args.dataset_code, args.version)
    print(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_code": args.dataset_code,
                "version": args.version,
                "status": "REVIEWED",
                "registered_documents": registered,
                "database_document_count": (
                    dataset["document_count"]
                    if dataset is not None
                    else registered
                ),
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
