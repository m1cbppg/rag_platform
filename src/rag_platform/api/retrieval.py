from fastapi import APIRouter

from src.rag_platform.application.retrieval_service import RetrievalService
from src.rag_platform.schemas.retrieval import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
async def retrieve_documents(
    request: RetrievalRequest,
) -> RetrievalResponse:
    """
    统一检索接口。

    支持：
    1. bm25
    2. vector
    3. hybrid

    注意：
    这里只返回召回文档，不做 Rerank，不生成答案。
    """

    service = RetrievalService()

    return await service.retrieve(request)