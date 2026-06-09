import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.corpus_models import (  # noqa: E402
    ReviewedDocumentOutcome,
)
from src.rag_platform.evaluation.corpus_persistence import (  # noqa: E402
    build_source_document_spec,
)
from src.rag_platform.evaluation.corpus_services import (  # noqa: E402
    CorpusFileStore,
    DocumentGenerationService,
    DocumentReviewService,
)
from src.rag_platform.evaluation.corpus_validation import (  # noqa: E402
    load_document_blueprints,
    validate_blueprint_plan,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.models import DatasetStatus  # noqa: E402
from src.rag_platform.infrastructure.dashscope_chat import (  # noqa: E402
    DashScopeChatClient,
)
from src.rag_platform.infrastructure.deepseek import DeepSeekClient  # noqa: E402


def load_review_documents(
    *,
    store: CorpusFileStore,
    all_blueprints: list,
    selected_blueprints: list,
) -> dict:
    required_codes = {
        item.source_doc_code
        for item in selected_blueprints
    }
    for blueprint in selected_blueprints:
        required_codes.update(blueprint.conflicts_with)
        if blueprint.supersedes:
            required_codes.add(blueprint.supersedes)
        if blueprint.version_group:
            required_codes.update(
                item.source_doc_code
                for item in all_blueprints
                if item.version_group == blueprint.version_group
            )

    documents = {}
    selected_codes = {
        item.source_doc_code
        for item in selected_blueprints
    }
    for code in required_codes:
        path = store.document_path(code)
        if not path.exists():
            if code in selected_codes:
                raise FileNotFoundError(f"缺少待审核源文档：{path}")
            continue
        documents[code] = store.load_document(code)
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用百炼Qwen审核M2源文档")
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
        "--manifest",
        type=Path,
        default=Path("evaluation/corpus/manifest.jsonl"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("evaluation/corpus/reviews"),
    )
    parser.add_argument(
        "--generate-prompt",
        type=Path,
        default=Path("evaluation/prompts/document_generate.txt"),
    )
    parser.add_argument(
        "--review-prompt",
        type=Path,
        default=Path("evaluation/prompts/document_review.txt"),
    )
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--code", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    blueprints = load_document_blueprints(args.blueprint)
    report = validate_blueprint_plan(blueprints)
    if not report.is_valid:
        raise ValueError("蓝图校验失败：" + "；".join(report.errors))

    selected_codes = set(args.code)
    selected = [
        item
        for item in blueprints
        if not selected_codes or item.source_doc_code in selected_codes
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    store = CorpusFileStore(args.source, args.manifest)
    documents = load_review_documents(
        store=store,
        all_blueprints=blueprints,
        selected_blueprints=selected,
    )

    deepseek = DeepSeekClient()
    qwen = DashScopeChatClient()
    generation_service = DocumentGenerationService(
        deepseek,
        args.generate_prompt.read_text(encoding="utf-8"),
    )
    review_service = DocumentReviewService(
        reviewer=qwen,
        generator=generation_service,
        prompt_template=args.review_prompt.read_text(encoding="utf-8"),
        max_regeneration_rounds=2,
    )
    repository = DatasetRepository() if args.dataset_id else None
    args.reviews.mkdir(parents=True, exist_ok=True)
    passed = 0
    rejected = 0
    skipped = 0

    try:
        for blueprint in selected:
            review_path = args.reviews / f"{blueprint.source_doc_code}.json"
            if review_path.exists() and not args.force:
                existing = ReviewedDocumentOutcome.model_validate_json(
                    review_path.read_text(encoding="utf-8")
                )
                if existing.review.passed:
                    if repository is not None:
                        source_path = store.document_path(
                            blueprint.source_doc_code
                        )
                        spec = build_source_document_spec(
                            blueprint=blueprint,
                            document=existing.document,
                            source_path=source_path,
                            outcome=existing,
                            project_root=PROJECT_ROOT,
                        )
                        repository.upsert_source_document(
                            args.dataset_id,
                            spec,
                        )
                    skipped += 1
                    print(f"SKIP {blueprint.source_doc_code}")
                    continue

            related_codes = {
                *blueprint.conflicts_with,
                *(
                    [blueprint.supersedes]
                    if blueprint.supersedes
                    else []
                ),
                *[
                    item.source_doc_code
                    for item in blueprints
                    if item.version_group
                    and item.version_group == blueprint.version_group
                    and item.source_doc_code != blueprint.source_doc_code
                ],
            }
            related_documents = [
                documents[code]
                for code in related_codes
                if code in documents
            ]
            outcome = await review_service.review(
                blueprint=blueprint,
                document=documents[blueprint.source_doc_code],
                related_documents=related_documents,
            )
            documents[blueprint.source_doc_code] = outcome.document
            if len(outcome.history) > 1:
                store.save_document(
                    outcome.document,
                    generation_round=len(outcome.history) - 1,
                    metadata={"reason": "qwen_review_regeneration"},
                )

            review_path.write_text(
                json.dumps(
                    outcome.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if repository is not None:
                source_path = store.document_path(blueprint.source_doc_code)
                spec = build_source_document_spec(
                    blueprint=blueprint,
                    document=outcome.document,
                    source_path=source_path,
                    outcome=outcome,
                    project_root=PROJECT_ROOT,
                )
                repository.upsert_source_document(args.dataset_id, spec)

            if outcome.review.passed:
                passed += 1
                print(
                    f"PASSED {blueprint.source_doc_code} "
                    f"score={outcome.review.overall_score:.2f}"
                )
            else:
                rejected += 1
                print(
                    f"REJECTED {blueprint.source_doc_code} "
                    f"issues={outcome.review.issues}"
                )
    finally:
        await qwen.aclose()
        await deepseek.aclose()

    all_passed = all(
        (args.reviews / f"{item.source_doc_code}.json").exists()
        and ReviewedDocumentOutcome.model_validate_json(
            (args.reviews / f"{item.source_doc_code}.json").read_text(
                encoding="utf-8"
            )
        ).review.passed
        for item in blueprints
    )
    if repository is not None and all_passed:
        repository.update_dataset_status(
            args.dataset_id,
            DatasetStatus.REVIEWED,
        )

    print(
        f"完成：通过{passed}篇，拒绝{rejected}篇，跳过{skipped}篇，"
        f"全部通过={all_passed}"
    )
    return 1 if rejected else 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
