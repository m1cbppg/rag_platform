from typing import Any, TypedDict


class RagState(TypedDict, total=False):
    """
    LangGraph RAG 工作流状态。

    State 是 LangGraph 中所有节点共享的数据结构。
    每个节点读取 State，并返回部分字段更新。
    """

    # --------------------
    # 请求基础信息
    # --------------------
    question: str
    session_id: str | None
    business_domain: str | None
    trace_id: str

    # --------------------
    # Query 理解结果
    # --------------------
    query_analysis: dict[str, Any]
    rewritten_question: str
    expanded_queries: list[str]
    retrieval_queries: list[str]
    retrieval_mode: str
    target_doc_types: list[str]

    # --------------------
    # 检索中间结果
    # --------------------
    retrieved_documents: list[dict[str, Any]]
    merged_documents: list[dict[str, Any]]

    # --------------------
    # 召回质量判断
    # --------------------
    retrieval_quality: dict[str, Any]
    need_rewrite: bool
    need_clarification: bool
    clarification_question: str | None

    # --------------------
    # Rerank 精排结果
    # --------------------
    reranked_documents: list[dict[str, Any]]
    rerank_info: dict[str, Any]

    # --------------------
    # Context 构建结果
    # --------------------
    context: str
    citations: list[dict[str, Any]]
    context_build_info: dict[str, Any]

    # --------------------
    # 后续模块预留
    # --------------------
    reranked_chunks: list[dict[str, Any]]
    context: str
    answer: str
    citations: list[dict[str, Any]]

    # --------------------
    # 工作流控制
    # --------------------
    current_node: str
    next_node: str
    status: str
    error: str | None

