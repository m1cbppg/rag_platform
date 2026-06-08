from pydantic import BaseModel, Field


class QueryAnalysisRequest(BaseModel):
    """
    Query 分析请求。

    这个接口只做 query 理解，不执行检索。
    """

    query: str = Field(..., description="用户原始问题")
    session_id: str | None = Field(default=None, description="会话ID")
    business_domain: str | None = Field(default=None, description="业务域，可选")


class QueryAnalysisResult(BaseModel):
    """
    Query 分析结果。

    这是模块 8 的核心数据结构。
    后续 LangGraph 会把它放到 RagState 中。
    """

    original_query: str = Field(..., description="用户原始问题")
    rewritten_query: str = Field(..., description="改写后的问题")

    expanded_queries: list[str] = Field(default_factory=list, description="扩展查询")
    target_doc_types: list[str] = Field(default_factory=list, description="目标文档类型：FAQ/SOP/RULE/MANUAL")

    retrieval_mode: str = Field(default="hybrid", description="检索模式：bm25/vector/hybrid")
    business_domain: str | None = Field(default=None, description="业务域")

    confidence: float = Field(default=0.0, description="置信度，0~1")
    reason: str = Field(default="", description="判断原因")

    need_clarification: bool = Field(default=False, description="是否需要澄清")
    clarification_question: str | None = Field(default=None, description="澄清问题")

    use_llm: bool = Field(default=False, description="是否使用LLM")
    fallback_used: bool = Field(default=False, description="是否使用兜底结果")


class QueryAnalysisResponse(BaseModel):
    """
    Query 分析接口响应。
    """

    trace_id: str
    result: QueryAnalysisResult