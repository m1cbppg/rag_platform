import pytest

from src.rag_platform.application.embedding_service import EmbeddingService
from src.rag_platform.application.search_index_service import SearchIndexService


class EmptyVectorRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def list_pending_embedding_tasks(
        self,
        limit: int,
        doc_id: int | None = None,
    ) -> list[dict]:
        self.calls.append((limit, doc_id))
        return []


class EmptySearchRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def list_pending_keyword_index_tasks(
        self,
        limit: int,
        doc_id: int | None = None,
    ) -> list[dict]:
        self.calls.append((limit, doc_id))
        return []


@pytest.mark.asyncio
async def test_embedding_tasks_can_be_run_for_one_document_only() -> None:
    repository = EmptyVectorRepository()
    service = EmbeddingService.__new__(EmbeddingService)
    service.vector_repository = repository

    response = await service.run_tasks(limit=50, doc_id=7001)

    assert response.success_count == 0
    assert repository.calls == [(50, 7001)]


def test_es_tasks_can_be_run_for_one_document_only() -> None:
    repository = EmptySearchRepository()
    service = SearchIndexService.__new__(SearchIndexService)
    service.repository = repository

    response = service.run_tasks(limit=50, doc_id=7001)

    assert response.success_count == 0
    assert repository.calls == [(50, 7001)]
