from langgraph.graph import StateGraph, END

from src.rag_platform.domain.rag_state import RagState


def build_rag_graph():
    """
    构建 RAG LangGraph 工作流。

    模块 1 先创建一个最小图：
    start -> placeholder -> END

    后续模块 9 会改成：
    analyze_query
    -> rewrite_query
    -> retrieve
    -> judge_retrieval
    -> rerank
    -> build_context
    -> generate_answer
    -> verify_citation
    -> save_log
    """

    graph = StateGraph(RagState)

    async def placeholder_node(state: RagState) -> RagState:
        """
        占位节点。

        LangGraph 的节点本质上就是一个函数：
        输入 state，返回更新后的 state。
        """

        state["answer"] = "LangGraph RAG 工作流占位节点已执行。"
        return state

    graph.add_node("placeholder", placeholder_node)

    graph.set_entry_point("placeholder")

    graph.add_edge("placeholder", END)

    return graph.compile()