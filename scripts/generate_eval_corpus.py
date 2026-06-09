import argparse
import asyncio
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.corpus_persistence import (  # noqa: E402
    build_source_document_spec,
)
from src.rag_platform.evaluation.corpus_services import (  # noqa: E402
    CorpusFileStore,
    DocumentGenerationService,
)
from src.rag_platform.evaluation.corpus_validation import (  # noqa: E402
    load_document_blueprints,
    validate_blueprint_plan,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.models import DatasetStatus  # noqa: E402
from src.rag_platform.infrastructure.deepseek import DeepSeekClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成M2受控电商售后源文档")
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path("evaluation/blueprints/ecommerce_document_plan.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/corpus/source"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/corpus/manifest.jsonl"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("evaluation/prompts/document_generate.txt"),
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
    if selected_codes - {item.source_doc_code for item in selected}:
        unknown = sorted(selected_codes - {item.source_doc_code for item in selected})
        raise ValueError(f"存在未知或被limit排除的文档编码：{unknown}")

    prompt_template = args.prompt.read_text(encoding="utf-8")
    store = CorpusFileStore(args.output, args.manifest)
    client = DeepSeekClient()
    service = DocumentGenerationService(client, prompt_template)
    repository = DatasetRepository() if args.dataset_id else None
    generated = 0
    skipped = 0

    try:
        for blueprint in selected:
            if not store.should_generate(blueprint.source_doc_code, args.force):
                skipped += 1
                print(f"SKIP {blueprint.source_doc_code}")
                continue

            document = await service.generate(blueprint)
            store.save_document(document, generation_round=0)
            if repository is not None:
                source_path = store.document_path(document.source_doc_code)
                spec = build_source_document_spec(
                    blueprint=blueprint,
                    document=document,
                    source_path=source_path,
                    project_root=PROJECT_ROOT,
                )
                repository.upsert_source_document(args.dataset_id, spec)
            generated += 1
            print(f"GENERATED {blueprint.source_doc_code}")
    finally:
        await client.aclose()

    all_generated = all(
        store.document_path(item.source_doc_code).exists()
        for item in blueprints
    )
    if repository is not None and all_generated:
        repository.update_dataset_status(
            args.dataset_id,
            DatasetStatus.GENERATED,
        )

    print(
        f"完成：生成{generated}篇，跳过{skipped}篇，"
        f"语料是否完整={all_generated}"
    )
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
