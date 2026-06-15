from typing import Any

from pydantic import BaseModel, Field


class ChatRequestV2(BaseModel):
    """
    正式 RAG Chat 请求。
    """

    question: str = Field(..., description="用户问题")
    session_id: str | None = Field(default=None, description="会话ID")
    business_domain: str | None = Field(default=None, description="业务域")
    top_k: int = Field(default=20, description="召回数量")


class ChatResponseV2(BaseModel):
    """
    正式 RAG Chat 响应。
    """

    trace_id: str
    answer_log_id: int | None = None

    question: str
    rewritten_question: str | None = None
    answer: str

    status: str
    action_decision: dict[str, Any] = Field(default_factory=dict)

    citations: list[dict[str, Any]] = Field(default_factory=list)
    citation_validation: dict[str, Any] = Field(default_factory=dict)

    retrieval_quality: dict[str, Any] = Field(default_factory=dict)
    rerank_info: dict[str, Any] = Field(default_factory=dict)
    context_build_info: dict[str, Any] = Field(default_factory=dict)
