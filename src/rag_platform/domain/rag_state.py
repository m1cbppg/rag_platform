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
    anchor_retrieval_queries: list[str]
    retrieval_mode: str
    target_doc_types: list[str]
    decomposition: dict[str, Any]
    retrieval_tasks: list[dict[str, Any]]
    sub_query_coverage: dict[str, Any]
    dependent_hop: dict[str, Any]

    # --------------------
    # 检索中间结果
    # --------------------
    top_k: int
    retrieved_documents: list[dict[str, Any]]
    merged_documents: list[dict[str, Any]]
    retrieval_round: int
    max_retrieval_rounds: int
    retrieval_attempts: list[dict[str, Any]]
    retry_strategy: str
    query_variant: str
    removed_filters: list[str]
    initial_business_domain: str | None
    initial_target_doc_types: list[str]
    final_retrieval_query: str

    # --------------------
    # 召回质量判断
    # --------------------
    retrieval_quality: dict[str, Any]
    quality_features: dict[str, Any]
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
    # 工作流控制
    # --------------------
    current_node: str
    next_node: str
    status: str
    error: str | None
