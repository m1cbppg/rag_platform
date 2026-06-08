from fastapi import APIRouter, Depends, Query

from src.rag_platform.application.embedding_service import EmbeddingService
from src.rag_platform.schemas.embedding import (
    EmbeddingTaskCreateResponse,
    EmbeddingTaskRunResponse,
)

router = APIRouter(prefix="/admin/embedding", tags=["embedding-admin"])


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@router.post("/tasks/create", response_model=EmbeddingTaskCreateResponse)
def create_embedding_tasks(
    doc_id: int | None = Query(default=None, description="文档ID，不传则扫描所有chunk"),
    limit: int = Query(default=100, description="本次最多创建多少个任务"),
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingTaskCreateResponse:
    """
    创建 embedding 任务。

    前置条件：
    文档已经完成模块 3 的 chunk 构建。
    """

    return service.create_tasks(
        doc_id=doc_id,
        limit=limit,
    )


@router.post("/tasks/run", response_model=EmbeddingTaskRunResponse)
async def run_embedding_tasks(
    limit: int = Query(default=50, description="本次最多执行多少个任务"),
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingTaskRunResponse:
    """
    执行 embedding 任务。

    会调用阿里 text-embedding-v4，并写入 Milvus。
    """

    return await service.run_tasks(limit=limit)
