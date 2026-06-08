from uuid import uuid4

from fastapi import APIRouter

from src.rag_platform.application.rerank_service import RerankService
from src.rag_platform.schemas.rerank import RerankTestRequest, RerankTestResponse

router = APIRouter(prefix="/admin/rerank", tags=["rerank-admin"])


@router.post("/test", response_model=RerankTestResponse)
async def test_rerank(
    request: RerankTestRequest,
) -> RerankTestResponse:
    """
    单独测试 qwen3-rerank。

    注意：
    正式 RAG 流程会通过 /rag/workflow/retrieval 调用 rerank。
    这个接口只是用于排查 qwen3-rerank 是否能正常工作。
    """

    trace_id = uuid4().hex
    service = RerankService()

    documents, rerank_info = await service.rerank_documents(
        trace_id=trace_id,
        query=request.query,
        documents=request.documents,
    )

    return RerankTestResponse(
        trace_id=trace_id,
        rerank_info=rerank_info,
        documents=documents,
    )