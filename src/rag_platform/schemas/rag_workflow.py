from typing import Any

from pydantic import BaseModel, Field


class RagRetrievalWorkflowRequest(BaseModel):
    """
    RAG 检索工作流请求。

    注意：
    这不是最终问答接口。
    它只执行到召回质量判断。
    """

    question: str = Field(..., description="用户问题")
    session_id: str | None = Field(default=None, description="会话ID")
    business_domain: str | None = Field(default=None, description="业务域")
    top_k: int = Field(default=10, description="召回数量")


class WorkflowDocumentResponse(BaseModel):
    chunk_id: int
    score: float | None = None
    source: str | None = None

    # 模块 10 新增
    rerank_score: float | None = None
    after_rank: int | None = None

    title: str | None = None
    title_path: str | None = None
    chunk_type: str | None = None
    business_domain: str | None = None
    source_section: str | None = None

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagRetrievalWorkflowResponse(BaseModel):
    trace_id: str
    question: str
    query_analysis: dict[str, Any] = Field(default_factory=dict)
    rewritten_question: str | None = None
    retrieval_mode: str | None = None
    target_doc_types: list[str] = Field(default_factory=list)
    decomposition: dict[str, Any] = Field(default_factory=dict)
    sub_query_coverage: dict[str, Any] = Field(
        default_factory=dict
    )
    dependent_hop: dict[str, Any] = Field(default_factory=dict)
    need_clarification: bool = False
    clarification_question: str | None = None
    status: str
    retrieval_quality: dict[str, Any] = Field(default_factory=dict)
    retrieval_round: int = 1
    max_retrieval_rounds: int = 1
    retrieval_attempts: list[dict[str, Any]] = Field(
        default_factory=list
    )

    # 召回阶段文档
    documents: list[WorkflowDocumentResponse] = Field(default_factory=list)

    # 模块 10 新增：精排阶段文档
    rerank_info: dict[str, Any] = Field(default_factory=dict)
    reranked_documents: list[WorkflowDocumentResponse] = Field(default_factory=list)

    context: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    context_build_info: dict[str, Any] = Field(default_factory=dict)
