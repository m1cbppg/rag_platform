import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.application.chunk_build_service import (  # noqa: E402
    ChunkBuildService,
)
from src.rag_platform.application.document_ingest_service import (  # noqa: E402
    DocumentIngestService,
)
from src.rag_platform.application.embedding_service import (  # noqa: E402
    EmbeddingService,
)
from src.rag_platform.application.search_index_service import (  # noqa: E402
    SearchIndexService,
)
from src.rag_platform.application.vector_collection_service import (  # noqa: E402
    VectorCollectionService,
)
from src.rag_platform.evaluation.corpus_ingest import (  # noqa: E402
    EvaluationCorpusIngestService,
    build_catalog_document,
)
from src.rag_platform.evaluation.corpus_models import (  # noqa: E402
    ReviewedDocumentOutcome,
)
from src.rag_platform.evaluation.corpus_validation import (  # noqa: E402
    load_document_blueprints,
    validate_blueprint_plan,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.infrastructure.repositories.document_repository import (  # noqa: E402
    DocumentRepository,
)


RENDERED_SUFFIX = {
    "FAQ": ".docx",
    "RULE": ".docx",
    "SOP": ".pdf",
    "MANUAL": ".pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行M3：导入评测语料并建立Chunk、Milvus和ES索引",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path("evaluation/blueprints/ecommerce_document_plan.json"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("evaluation/corpus/reviews"),
    )
    parser.add_argument(
        "--rendered",
        type=Path,
        default=Path("evaluation/corpus/rendered"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("evaluation/corpus/catalog.json"),
    )
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--dataset-code", default="rag_eval_ecommerce")
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="只处理指定source_doc_code，可重复传入",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="只处理排序后的前N篇，主要用于联调",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=1000,
        help="每篇文档最多处理的Chunk任务数",
    )
    return parser.parse_args()


def _resolve_dataset(
    repository: DatasetRepository,
    *,
    dataset_id: int | None,
    dataset_code: str,
    version: str,
) -> dict:
    dataset = repository.find_dataset(dataset_code, version)
    if dataset_id is None:
        if dataset is None:
            raise ValueError(
                f"数据集不存在：dataset_code={dataset_code}, version={version}"
            )
        return dataset

    if dataset is None or int(dataset["id"]) != dataset_id:
        raise ValueError(
            f"dataset_id={dataset_id} 与 {dataset_code}/{version} 不匹配"
        )
    return dataset


def _load_passed_outcome(
    reviews_dir: Path,
    source_doc_code: str,
) -> ReviewedDocumentOutcome:
    review_path = reviews_dir / f"{source_doc_code}.json"
    if not review_path.exists():
        raise FileNotFoundError(f"缺少审核结果：{review_path}")
    outcome = ReviewedDocumentOutcome.model_validate_json(
        review_path.read_text(encoding="utf-8")
    )
    if not outcome.review.passed:
        raise ValueError(f"{source_doc_code} 未通过M2审核，禁止导入")
    return outcome


def _rendered_path(rendered_dir: Path, blueprint) -> Path:
    suffix = RENDERED_SUFFIX[blueprint.doc_type.value]
    path = rendered_dir / f"{blueprint.source_doc_code}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"缺少渲染文档：{path}")
    return path


def _select_blueprints(blueprints: list, args: argparse.Namespace) -> list:
    selected = blueprints
    if args.only:
        requested = set(args.only)
        known = {item.source_doc_code for item in blueprints}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"--only 包含未知文档：{unknown}")
        selected = [
            item for item in blueprints if item.source_doc_code in requested
        ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于0")
        selected = selected[:args.limit]
    return selected


def _export_catalog(
    *,
    catalog_path: Path,
    dataset: dict,
    blueprints: list,
    reviews_dir: Path,
    rendered_dir: Path,
    dataset_repository: DatasetRepository,
    document_repository: DocumentRepository,
    require_complete: bool,
) -> dict:
    blueprint_by_code = {
        blueprint.source_doc_code: blueprint for blueprint in blueprints
    }
    source_rows = dataset_repository.list_source_documents(int(dataset["id"]))
    source_by_code = {row["source_doc_code"]: row for row in source_rows}
    missing_registrations = sorted(set(blueprint_by_code) - set(source_by_code))
    if missing_registrations:
        raise RuntimeError(
            f"评测源文档尚未注册到MySQL：{missing_registrations}"
        )

    catalog_documents = []
    unmapped_codes = []
    for source_doc_code in sorted(blueprint_by_code):
        row = source_by_code[source_doc_code]
        mapped_doc_id = row.get("mapped_doc_id")
        if mapped_doc_id is None:
            unmapped_codes.append(source_doc_code)
            continue

        blueprint = blueprint_by_code[source_doc_code]
        outcome = _load_passed_outcome(reviews_dir, source_doc_code)
        rendered_path = _rendered_path(rendered_dir, blueprint)
        chunks = document_repository.list_chunks_by_doc_id(int(mapped_doc_id))
        if not chunks:
            raise RuntimeError(f"{source_doc_code} 已映射但没有ACTIVE Chunk")
        catalog_documents.append(
            build_catalog_document(
                blueprint=blueprint,
                document=outcome.document,
                mapped_doc_id=int(mapped_doc_id),
                chunks=chunks,
                source_content_sha256=row["source_content_sha256"],
                rendered_path=rendered_path,
            )
        )

    if require_complete and unmapped_codes:
        raise RuntimeError(f"仍有文档未完成M3映射：{unmapped_codes}")

    catalog = {
        "schema_version": "1.0",
        "dataset_id": int(dataset["id"]),
        "dataset_code": dataset["dataset_code"],
        "dataset_version": dataset["version"],
        "document_count": len(catalog_documents),
        "documents": catalog_documents,
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


async def run(args: argparse.Namespace) -> int:
    blueprints = load_document_blueprints(args.blueprint)
    report = validate_blueprint_plan(blueprints)
    if not report.is_valid:
        raise ValueError("蓝图校验失败：" + "；".join(report.errors))
    selected_blueprints = _select_blueprints(blueprints, args)

    dataset_repository = DatasetRepository()
    dataset = _resolve_dataset(
        dataset_repository,
        dataset_id=args.dataset_id,
        dataset_code=args.dataset_code,
        version=args.version,
    )
    source_rows = dataset_repository.list_source_documents(int(dataset["id"]))
    registered_codes = {row["source_doc_code"] for row in source_rows}
    missing = sorted(
        blueprint.source_doc_code
        for blueprint in selected_blueprints
        if blueprint.source_doc_code not in registered_codes
    )
    if missing:
        raise RuntimeError(f"请先执行M2注册，缺少：{missing}")

    vector_init = VectorCollectionService().init_collection()
    search_service = SearchIndexService()
    search_init = search_service.init_index()
    document_repository = DocumentRepository()
    ingest_service = EvaluationCorpusIngestService(
        ingest_service=DocumentIngestService(),
        chunk_service=ChunkBuildService(),
        embedding_service=EmbeddingService(),
        search_service=search_service,
        document_repository=document_repository,
        dataset_repository=dataset_repository,
        task_limit=args.task_limit,
    )

    results = []
    total = len(selected_blueprints)
    for index, blueprint in enumerate(selected_blueprints, start=1):
        outcome = _load_passed_outcome(
            args.reviews,
            blueprint.source_doc_code,
        )
        rendered_path = _rendered_path(args.rendered, blueprint)
        result = await ingest_service.ingest_document(
            dataset_id=int(dataset["id"]),
            blueprint=blueprint,
            document=outcome.document,
            rendered_path=rendered_path,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{total}",
                    "source_doc_code": result.source_doc_code,
                    "mapped_doc_id": result.mapped_doc_id,
                    "chunk_count": len(result.chunk_ids),
                    "embedding": result.embedding_task_summary,
                    "elasticsearch": result.keyword_task_summary,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    full_run = not args.only and args.limit is None
    catalog = _export_catalog(
        catalog_path=args.catalog,
        dataset=dataset,
        blueprints=blueprints,
        reviews_dir=args.reviews,
        rendered_dir=args.rendered,
        dataset_repository=dataset_repository,
        document_repository=document_repository,
        require_complete=full_run,
    )
    print(
        json.dumps(
            {
                "dataset_id": int(dataset["id"]),
                "processed_documents": len(results),
                "catalog_documents": catalog["document_count"],
                "catalog_path": args.catalog.as_posix(),
                "milvus_collection": vector_init.collection_name,
                "elasticsearch_index": search_init.index_name,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
