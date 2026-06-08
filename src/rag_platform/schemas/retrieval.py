from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """
    检索请求。

    mode:
        auto：先做 Query 理解，然后自动选择 bm25/vector/hybrid
        bm25：只走 ES BM25
        vector：只走 Milvus 向量
        hybrid：BM25 + Vector 融合
    """

    query: str = Field(..., description="用户问题")
    mode: str = Field(default="auto", description="检索模式：auto/bm25/vector/hybrid")
    top_k: int = Field(default=10, description="最终返回数量")

    doc_type: str | None = Field(default=None, description="文档类型过滤：FAQ/SOP/RULE/MANUAL")
    business_domain: str | None = Field(default=None, description="业务域过滤")


class RetrievedDocumentResponse(BaseModel):
    """
    单个召回文档响应。
    """

    chunk_id: int
    score: float | None = None
    source: str | None = None
    title: str | None = None
    title_path: str | None = None
    chunk_type: str | None = None
    business_domain: str | None = None
    source_section: str | None = None
    content: str


class RetrievalResponse(BaseModel):
    """
    检索响应。
    """

    query: str
    mode: str
    documents: list[RetrievedDocumentResponse]