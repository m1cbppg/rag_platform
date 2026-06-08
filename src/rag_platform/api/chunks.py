from fastapi import APIRouter, Depends, Path

from src.rag_platform.application.chunk_build_service import ChunkBuildService
from src.rag_platform.schemas.chunk import ChunkBuildResponse

router = APIRouter(prefix="/documents", tags=["chunks"])


def get_chunk_build_service() -> ChunkBuildService:
    return ChunkBuildService()


@router.post("/{doc_id}/chunks/build", response_model=ChunkBuildResponse)
def build_document_chunks(
    doc_id: int = Path(..., description="文档ID"),
    service: ChunkBuildService = Depends(get_chunk_build_service),
) -> ChunkBuildResponse:
    """
    为指定文档构建 chunk。

    前置条件：
    文档必须已经完成模块 2 的解析和清洗，即状态为 CLEANED。
    """

    return service.build_chunks_for_document(doc_id)
