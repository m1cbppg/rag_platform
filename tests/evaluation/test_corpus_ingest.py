from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag_platform.evaluation.corpus_ingest import (
    EvaluationCorpusIngestService,
    build_catalog_document,
)
from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    GeneratedSourceDocument,
)


def _blueprint() -> DocumentBlueprint:
    return DocumentBlueprint(
        source_doc_code="FAQ_ORDER_001",
        doc_type="FAQ",
        title="订单取消FAQ",
        topic="order",
        version="1.0",
        required_facts=[
            {"fact_key": "cancel_pending", "description": "待支付订单可取消"},
        ],
        required_identifiers=["F-ORDER-001"],
        required_sections=["待支付订单"],
    )


def _document() -> GeneratedSourceDocument:
    return GeneratedSourceDocument.model_validate(
        {
            "source_doc_code": "FAQ_ORDER_001",
            "title": "订单取消FAQ",
            "doc_type": "FAQ",
            "topic": "order",
            "version": "1.0",
            "sections": [
                {
                    "section_code": "Q1",
                    "heading": "待支付订单",
                    "content": "F-ORDER-001 待支付订单可以直接取消。",
                    "aliases": ["怎么取消？", "订单不要了？"],
                    "facts": [
                        {
                            "fact_key": "cancel_pending",
                            "fact_text": "待支付订单可以直接取消",
                        }
                    ],
                }
            ],
        }
    )


class FakeIngestService:
    def __init__(self, status: str = "CLEANED") -> None:
        self.status = status
        self.calls: list[Path] = []

    def ingest_file_path(self, file_path: Path, **kwargs):
        self.calls.append(file_path)
        return SimpleNamespace(doc_id=81, status=self.status)


class FakeChunkService:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def build_chunks_for_document(self, doc_id: int):
        self.calls.append(doc_id)
        return SimpleNamespace(status="CHUNKED", chunk_count=1)


class FakeEmbeddingService:
    def __init__(self, failed_count: int = 0) -> None:
        self.failed_count = failed_count
        self.run_calls: list[tuple[int, int]] = []

    def create_tasks(self, doc_id: int, limit: int):
        return SimpleNamespace(task_count=1)

    async def run_tasks(self, limit: int, doc_id: int | None = None):
        self.run_calls.append((limit, doc_id))
        return SimpleNamespace(
            success_count=1 - self.failed_count,
            failed_count=self.failed_count,
        )

    def get_task_summary(self, doc_id: int):
        return {"SUCCESS": 1} if not self.failed_count else {"FAILED": 1}


class FakeSearchService:
    def __init__(self) -> None:
        self.run_calls: list[tuple[int, int]] = []

    def create_tasks(self, doc_id: int, limit: int):
        return SimpleNamespace(task_count=1)

    def run_tasks(self, limit: int, doc_id: int | None = None):
        self.run_calls.append((limit, doc_id))
        return SimpleNamespace(success_count=1, failed_count=0)

    def get_task_summary(self, doc_id: int):
        return {"SUCCESS": 1}


class FakeDocumentRepository:
    def list_chunks_by_doc_id(self, doc_id: int) -> list[dict]:
        return [
            {
                "id": 901,
                "doc_id": doc_id,
                "title": "待支付订单",
                "title_path": "FAQ > 待支付订单",
                "content": (
                    "问题：如何取消待支付订单？\n"
                    "答案：F-ORDER-001 待支付订单可以直接取消"
                ),
                "summary": "待支付订单可以直接取消",
                "source_section": "FAQ-1",
                "sort_order": 1,
            }
        ]


class FakeDatasetRepository:
    def __init__(self) -> None:
        self.mappings: list[tuple[int, str, int]] = []

    def map_source_document(
        self,
        dataset_id: int,
        source_doc_code: str,
        mapped_doc_id: int,
    ) -> None:
        self.mappings.append((dataset_id, source_doc_code, mapped_doc_id))


@pytest.mark.asyncio
async def test_ingest_maps_document_only_after_chunk_and_indexes_succeed(
    tmp_path: Path,
) -> None:
    rendered_path = tmp_path / "FAQ_ORDER_001.docx"
    rendered_path.write_bytes(b"docx")
    dataset_repository = FakeDatasetRepository()
    chunk_service = FakeChunkService()
    embedding_service = FakeEmbeddingService()
    search_service = FakeSearchService()
    service = EvaluationCorpusIngestService(
        ingest_service=FakeIngestService(),
        chunk_service=chunk_service,
        embedding_service=embedding_service,
        search_service=search_service,
        document_repository=FakeDocumentRepository(),
        dataset_repository=dataset_repository,
    )

    result = await service.ingest_document(
        dataset_id=7,
        blueprint=_blueprint(),
        document=_document(),
        rendered_path=rendered_path,
    )

    assert result.mapped_doc_id == 81
    assert result.chunk_ids == [901]
    assert chunk_service.calls == [81]
    assert embedding_service.run_calls == [(1000, 81)]
    assert search_service.run_calls == [(1000, 81)]
    assert dataset_repository.mappings == [(7, "FAQ_ORDER_001", 81)]


@pytest.mark.asyncio
async def test_ingest_does_not_rebuild_chunks_for_existing_chunked_document(
    tmp_path: Path,
) -> None:
    rendered_path = tmp_path / "FAQ_ORDER_001.docx"
    rendered_path.write_bytes(b"docx")
    chunk_service = FakeChunkService()
    service = EvaluationCorpusIngestService(
        ingest_service=FakeIngestService(status="CHUNKED"),
        chunk_service=chunk_service,
        embedding_service=FakeEmbeddingService(),
        search_service=FakeSearchService(),
        document_repository=FakeDocumentRepository(),
        dataset_repository=FakeDatasetRepository(),
    )

    await service.ingest_document(
        dataset_id=7,
        blueprint=_blueprint(),
        document=_document(),
        rendered_path=rendered_path,
    )

    assert chunk_service.calls == []


@pytest.mark.asyncio
async def test_ingest_does_not_map_when_embedding_fails(tmp_path: Path) -> None:
    rendered_path = tmp_path / "FAQ_ORDER_001.docx"
    rendered_path.write_bytes(b"docx")
    dataset_repository = FakeDatasetRepository()
    service = EvaluationCorpusIngestService(
        ingest_service=FakeIngestService(),
        chunk_service=FakeChunkService(),
        embedding_service=FakeEmbeddingService(failed_count=1),
        search_service=FakeSearchService(),
        document_repository=FakeDocumentRepository(),
        dataset_repository=dataset_repository,
    )

    with pytest.raises(RuntimeError, match="Embedding"):
        await service.ingest_document(
            dataset_id=7,
            blueprint=_blueprint(),
            document=_document(),
            rendered_path=rendered_path,
        )

    assert dataset_repository.mappings == []


def test_catalog_maps_fact_key_to_matching_chunk() -> None:
    catalog = build_catalog_document(
        blueprint=_blueprint(),
        document=_document(),
        mapped_doc_id=81,
        chunks=FakeDocumentRepository().list_chunks_by_doc_id(81),
        source_content_sha256="a" * 64,
    )

    assert catalog["source_doc_code"] == "FAQ_ORDER_001"
    assert catalog["mapped_doc_id"] == 81
    assert catalog["chunk_ids"] == [901]
    assert catalog["facts"] == [
        {
            "fact_key": "cancel_pending",
            "fact_text": "待支付订单可以直接取消",
            "chunk_ids": [901],
        }
    ]
