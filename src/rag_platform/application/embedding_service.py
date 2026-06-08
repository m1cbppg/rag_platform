from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.vector import EmbeddingTaskStatus
from src.rag_platform.infrastructure.dashscope_embedding import DashScopeEmbeddingClient
from src.rag_platform.infrastructure.milvus_vector_store import MilvusVectorStore
from src.rag_platform.infrastructure.repositories.vector_repository import VectorRepository
from src.rag_platform.rag.embeddings.embedding_text_builder import EmbeddingTextBuilder
from src.rag_platform.schemas.embedding import (
    EmbeddingTaskCreateResponse,
    EmbeddingTaskRunResponse,
)


class EmbeddingService:
    """
    Embedding 应用服务。

    负责：
    1. 为 chunk 创建 embedding task；
    2. 执行 embedding task；
    3. 写入 Milvus；
    4. 更新任务状态。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_repository = VectorRepository()
        self.text_builder = EmbeddingTextBuilder()
        self.embedding_client = DashScopeEmbeddingClient()
        self.vector_store = MilvusVectorStore()

    def create_tasks(
        self,
        doc_id: int | None,
        limit: int,
    ) -> EmbeddingTaskCreateResponse:
        """
        为 chunk 创建 embedding 任务。

        doc_id:
            如果传入，只为指定文档创建任务；
            如果不传，则扫描所有 ACTIVE chunk。

        limit:
            限制本次最多扫描多少个 chunk。
        """

        chunks = self.vector_repository.list_active_chunks_for_embedding(
            doc_id=doc_id,
            limit=limit,
        )

        task_count = 0

        for chunk in chunks:
            embedding_text = self.text_builder.build_embedding_text(chunk)
            embedding_text_hash = self.text_builder.hash_text(embedding_text)

            self.vector_repository.upsert_embedding_task(
                chunk_id=chunk["id"],
                doc_id=chunk["doc_id"],
                embedding_model=self.settings.embedding_model,
                embedding_dimension=self.settings.embedding_dimension,
                embedding_output_type=self.settings.embedding_output_type,
                embedding_text_hash=embedding_text_hash,
                milvus_collection=self.settings.milvus_collection,
            )

            task_count += 1

        return EmbeddingTaskCreateResponse(
            task_count=task_count,
            message="embedding 任务创建完成",
        )

    async def run_tasks(self, limit: int) -> EmbeddingTaskRunResponse:
        """
        执行 embedding 任务。

        这里按 batch 调用阿里 text-embedding-v4。
        """

        tasks = self.vector_repository.list_pending_embedding_tasks(limit=limit)

        if not tasks:
            return EmbeddingTaskRunResponse(
                success_count=0,
                failed_count=0,
                message="没有待执行的 embedding 任务",
            )

        batch_size = self.settings.embedding_batch_size
        success_count = 0
        failed_count = 0

        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]

            try:
                success = await self._run_one_batch(batch)
                success_count += success

            except Exception as exc:
                failed_count += len(batch)

                for task in batch:
                    self.vector_repository.update_embedding_task_status(
                        task_id=task["task_id"],
                        status=EmbeddingTaskStatus.FAILED.value,
                        error_message=str(exc),
                        increase_retry=True,
                    )

        return EmbeddingTaskRunResponse(
            success_count=success_count,
            failed_count=failed_count,
            message="embedding 任务执行完成",
        )

    async def _run_one_batch(self, tasks: list[dict]) -> int:
        """
        执行一个 batch。

        步骤：
        1. 将任务状态改为 PROCESSING；
        2. 构建 embedding_text；
        3. 调用 DashScope；
        4. 写入 Milvus；
        5. 更新任务状态为 SUCCESS。
        """

        for task in tasks:
            self.vector_repository.update_embedding_task_status(
                task_id=task["task_id"],
                status=EmbeddingTaskStatus.PROCESSING.value,
            )

        embedding_texts = [
            self.text_builder.build_embedding_text(task)
            for task in tasks
        ]

        vectors = await self.embedding_client.embed_documents(embedding_texts)

        milvus_rows = []

        for task, vector in zip(tasks, vectors):
            milvus_rows.append({
                "chunk_id": task["chunk_id"],
                "doc_id": task["doc_id"],
                "chunk_code": task["chunk_code"],
                "doc_type": task["chunk_type"],
                "business_domain": task.get("business_domain") or "",
                "version": task.get("version") or "",
                "status": "ACTIVE",
                "embedding": vector,
            })

        self.vector_store.upsert_vectors(milvus_rows)

        for task in tasks:
            self.vector_repository.update_embedding_task_status(
                task_id=task["task_id"],
                status=EmbeddingTaskStatus.SUCCESS.value,
                milvus_pk=task["chunk_id"],
                error_message=None,
            )

        return len(tasks)