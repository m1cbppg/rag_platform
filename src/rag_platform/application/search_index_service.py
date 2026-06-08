from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.search import KeywordIndexTaskStatus
from src.rag_platform.infrastructure.elasticsearch_index import ElasticsearchIndexManager
from src.rag_platform.infrastructure.elasticsearch_store import ElasticsearchChunkStore
from src.rag_platform.infrastructure.repositories.search_repository import SearchRepository
from src.rag_platform.schemas.search import (
    KeywordIndexTaskCreateResponse,
    KeywordIndexTaskRunResponse,
    SearchIndexInitResponse,
)


class SearchIndexService:
    """
    ES BM25 索引服务。

    负责：
    1. 初始化 ES index；
    2. 创建 chunk 索引任务；
    3. 执行索引任务；
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_manager = ElasticsearchIndexManager()
        self.chunk_store = ElasticsearchChunkStore()
        self.repository = SearchRepository()

    def init_index(self) -> SearchIndexInitResponse:
        try:
            self.index_manager.create_index_if_not_exists()

            self.repository.upsert_index_state(
                search_engine="elasticsearch",
                index_name=self.settings.es_chunk_index,
                analyzer=self.settings.es_analyzer,
                search_analyzer=self.settings.es_search_analyzer,
                status="CREATED",
                error_message=None,
            )

            return SearchIndexInitResponse(
                index_name=self.settings.es_chunk_index,
                status="CREATED",
                message="ES BM25 index 初始化完成",
            )

        except Exception as exc:
            self.repository.upsert_index_state(
                search_engine="elasticsearch",
                index_name=self.settings.es_chunk_index,
                analyzer=self.settings.es_analyzer,
                search_analyzer=self.settings.es_search_analyzer,
                status="FAILED",
                error_message=str(exc),
            )
            raise

    def create_tasks(
        self,
        doc_id: int | None,
        limit: int,
    ) -> KeywordIndexTaskCreateResponse:
        chunks = self.repository.list_active_chunks_for_indexing(
            doc_id=doc_id,
            limit=limit,
        )

        task_count = 0

        for chunk in chunks:
            index_text = self.repository.build_index_text(chunk)
            index_text_hash = self.repository.hash_index_text(index_text)

            self.repository.upsert_keyword_index_task(
                chunk_id=chunk["id"],
                doc_id=chunk["doc_id"],
                index_name=self.settings.es_chunk_index,
                index_text_hash=index_text_hash,
            )

            task_count += 1

        return KeywordIndexTaskCreateResponse(
            task_count=task_count,
            message="ES 索引任务创建完成",
        )

    def run_tasks(self, limit: int) -> KeywordIndexTaskRunResponse:
        tasks = self.repository.list_pending_keyword_index_tasks(limit=limit)

        if not tasks:
            return KeywordIndexTaskRunResponse(
                success_count=0,
                failed_count=0,
                message="没有待执行的 ES 索引任务",
            )

        success_count = 0
        failed_count = 0

        batch_size = self.settings.es_index_batch_size

        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]

            try:
                self._run_one_batch(batch)
                success_count += len(batch)

            except Exception as exc:
                failed_count += len(batch)

                for task in batch:
                    self.repository.update_keyword_index_task_status(
                        task_id=task["task_id"],
                        status=KeywordIndexTaskStatus.FAILED.value,
                        error_message=str(exc),
                        increase_retry=True,
                    )

        return KeywordIndexTaskRunResponse(
            success_count=success_count,
            failed_count=failed_count,
            message="ES 索引任务执行完成",
        )

    def _run_one_batch(self, tasks: list[dict]) -> None:
        for task in tasks:
            self.repository.update_keyword_index_task_status(
                task_id=task["task_id"],
                status=KeywordIndexTaskStatus.PROCESSING.value,
            )

        es_docs = []

        for task in tasks:
            es_docs.append({
                "chunk_id": task["chunk_id"],
                "doc_id": task["doc_id"],
                "chunk_code": task["chunk_code"],
                "chunk_type": task["chunk_type"],
                "title": task.get("title") or "",
                "title_path": task.get("title_path") or "",
                "content": task.get("content") or "",
                "keywords": task.get("keywords") or "",
                "tags": task.get("tags") or "",
                "business_domain": task.get("business_domain") or "",
                "version": task.get("version") or "",
                "source_section": task.get("source_section") or "",
                "status": "ACTIVE",
            })

        self.chunk_store.bulk_index_chunks(es_docs)

        for task in tasks:
            self.repository.update_keyword_index_task_status(
                task_id=task["task_id"],
                status=KeywordIndexTaskStatus.SUCCESS.value,
                error_message=None,
            )