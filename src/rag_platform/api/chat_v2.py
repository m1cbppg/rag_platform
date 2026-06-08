from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.rag_platform.application.chat_service import ChatService
from src.rag_platform.schemas.chat_v2 import ChatRequestV2, ChatResponseV2

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponseV2)
async def chat(
    request: ChatRequestV2,
) -> ChatResponseV2:
    """
    非流式 RAG Chat 接口。

    流程：
    1. RAG 检索工作流；
    2. Context 构建；
    3. DeepSeek 生成答案；
    4. 返回完整 JSON。
    """

    service = ChatService()
    return await service.chat(request)


@router.post("/stream")
async def chat_stream(
    request: ChatRequestV2,
) -> StreamingResponse:
    """
    SSE 流式 RAG Chat 接口。

    返回事件：
    - trace
    - retrieval
    - context
    - delta
    - done
    - error
    """

    service = ChatService()

    return StreamingResponse(
        service.stream_chat(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )