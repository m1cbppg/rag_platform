from pydantic import BaseModel, Field

from src.rag_platform.schemas.chat_v2 import ChatResponseV2
from src.rag_platform.schemas.rag_workflow import (
    RagRetrievalWorkflowResponse,
)


class ChatExecutionResult(BaseModel):
    response: ChatResponseV2
    workflow: RagRetrievalWorkflowResponse
    latency_ms: int = Field(ge=0)
