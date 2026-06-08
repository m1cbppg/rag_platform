from fastapi import APIRouter

from src.rag_platform.application.rag_workflow_service import RagWorkflowService
from src.rag_platform.schemas.rag_workflow import (
    RagRetrievalWorkflowRequest,
    RagRetrievalWorkflowResponse,
)

router = APIRouter(prefix="/rag/workflow", tags=["rag-workflow"])


@router.post("/retrieval", response_model=RagRetrievalWorkflowResponse)
async def run_retrieval_workflow(
    request: RagRetrievalWorkflowRequest,
) -> RagRetrievalWorkflowResponse:
    """
    执行 RAG 检索工作流。

    当前流程：
    1. Query 理解；
    2. 条件路由；
    3. BM25 / Vector / Hybrid 检索；
    4. 文档去重合并；
    5. 召回质量判断。

    注意：
    当前不做答案生成。
    """

    service = RagWorkflowService()

    return await service.run_retrieval_workflow(request)