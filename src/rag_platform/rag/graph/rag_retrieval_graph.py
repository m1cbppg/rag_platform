from langgraph.graph import END, START, StateGraph

from src.rag_platform.application.query_understanding_service import QueryUnderstandingService
from src.rag_platform.domain.rag_state import RagState
from src.rag_platform.rag.graph.document_codec import GraphDocumentCodec
from src.rag_platform.rag.graph.retrieval_quality import RetrievalQualityJudge
from src.rag_platform.rag.retrievers.langchain_bm25_retriever import LangChainBM25Retriever
from src.rag_platform.rag.retrievers.langchain_hybrid_retriever import LangChainHybridRetriever
from src.rag_platform.rag.retrievers.langchain_vector_retriever import LangChainVectorRetriever
from src.rag_platform.schemas.query_analysis import QueryAnalysisRequest
from src.rag_platform.application.rerank_service import RerankService
from src.rag_platform.application.context_build_service import ContextBuildService


class RagRetrievalGraphBuilder:
    """
    RAG 检索工作流构建器。

    这个 graph 当前只负责：
    1. Query 理解；
    2. 检索路由；
    3. Retriever 调用；
    4. 文档合并；
    5. 召回质量判断。

    后续模块会继续接：
    - Rerank
    - ContextBuilder
    - AnswerGenerator
    """

    def __init__(
        self,
        query_understanding_service=None,
        rerank_service=None,
        context_build_service=None,
        bm25_retriever_factory=None,
        vector_retriever_factory=None,
        hybrid_retriever_factory=None,
        document_codec=None,
        quality_judge=None,
    ) -> None:
        self.query_understanding_service = (
            query_understanding_service or QueryUnderstandingService()
        )
        self.rerank_service = rerank_service or RerankService()
        self.context_build_service = context_build_service or ContextBuildService()
        self.bm25_retriever_factory = (
            bm25_retriever_factory or LangChainBM25Retriever
        )
        self.vector_retriever_factory = (
            vector_retriever_factory or LangChainVectorRetriever
        )
        self.hybrid_retriever_factory = (
            hybrid_retriever_factory or LangChainHybridRetriever
        )
        self.document_codec = document_codec or GraphDocumentCodec()
        self.quality_judge = quality_judge or RetrievalQualityJudge()

    def build(self):
        """
        构建并编译 LangGraph。

        StateGraph 是 builder，添加节点和边之后，需要 compile() 成可执行 graph。
        """

        graph = StateGraph(RagState)

        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("retrieve_bm25", self.retrieve_bm25)
        graph.add_node("retrieve_vector", self.retrieve_vector)
        graph.add_node("retrieve_hybrid", self.retrieve_hybrid)
        graph.add_node("merge_documents", self.merge_documents)
        graph.add_node("judge_retrieval_quality", self.judge_retrieval_quality)
        graph.add_node("rerank_documents", self.rerank_documents)
        graph.add_node("build_context", self.build_context)
        graph.add_node("finish", self.finish)

        graph.add_edge(START, "analyze_query")

        graph.add_conditional_edges(
            "analyze_query",
            self.route_after_query_analysis,
            {
                "bm25": "retrieve_bm25",
                "vector": "retrieve_vector",
                "hybrid": "retrieve_hybrid",
            },
        )

        graph.add_edge("retrieve_bm25", "merge_documents")
        graph.add_edge("retrieve_vector", "merge_documents")
        graph.add_edge("retrieve_hybrid", "merge_documents")

        graph.add_edge("merge_documents", "judge_retrieval_quality")
        graph.add_conditional_edges(
            "judge_retrieval_quality",
            self.route_after_quality_judge,
            {
                "rerank": "rerank_documents",
                "context": "build_context",
                "finish": "finish",
            },
        )

        graph.add_edge("rerank_documents", "build_context")
        graph.add_edge("build_context", "finish")
        graph.add_edge("finish", END)

        return graph.compile()

    def build_context(self, state: RagState) -> dict:
        """
        节点：Context 构建。

        优先使用 reranked_documents。
        如果 reranked_documents 为空，则 fallback 到 merged_documents。
        """

        trace_id = state.get("trace_id") or ""
        query = state.get("rewritten_question") or state.get("question") or ""

        documents = state.get("reranked_documents") or state.get("merged_documents") or []

        result, context_build_info = self.context_build_service.build_context(
            trace_id=trace_id,
            query_text=query,
            documents=documents,
        )

        citations = [
            {
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "doc_id": citation.doc_id,
                "title": citation.title,
                "title_path": citation.title_path,
                "source_section": citation.source_section,
                "chunk_type": citation.chunk_type,
                "expansion_type": citation.expansion_type,
                "sort_order": citation.sort_order,
            }
            for citation in result.citations
        ]

        return {
            "context": result.context,
            "citations": citations,
            "context_build_info": context_build_info,
            "current_node": "build_context",
            "status": "CONTEXT_READY",
        }

    def route_after_quality_judge(self, state: RagState) -> str:
        """
        质量判断后的条件路由。

        POOR：没有候选，结束。
        GOOD/WEAK：进入 rerank。
        后续关闭 rerank 时，可以直接进入 context。
        """

        quality = state.get("retrieval_quality", {})
        quality_level = quality.get("quality")

        if quality_level == "POOR":
            return "finish"

        return "rerank"

    async def rerank_documents(self, state: RagState) -> dict:
        """
        节点 5：qwen3-rerank 精排。

        输入：
            merged_documents

        输出：
            reranked_documents
            rerank_info
        """

        trace_id = state.get("trace_id") or ""
        query = state.get("rewritten_question") or state.get("question") or ""
        documents = state.get("merged_documents", [])

        reranked_documents, rerank_info = await self.rerank_service.rerank_documents(
            trace_id=trace_id,
            query=query,
            documents=documents,
        )

        return {
            "reranked_documents": reranked_documents,
            "rerank_info": rerank_info,
            "current_node": "rerank_documents",
            "status": "RERANKED",
        }

    async def analyze_query(self, state: RagState) -> dict:
        """
        节点 1：Query 理解。

        输入：
            state["question"]

        输出：
            query_analysis
            rewritten_question
            expanded_queries
            retrieval_queries
            retrieval_mode
            target_doc_types
        """

        question = state["question"]
        business_domain = state.get("business_domain")
        session_id = state.get("session_id")

        response = await self.query_understanding_service.analyze(
            QueryAnalysisRequest(
                query=question,
                session_id=session_id,
                business_domain=business_domain,
            )
        )

        result = response.result

        retrieval_queries = self._build_retrieval_queries(
            rewritten_query=result.rewritten_query,
            expanded_queries=result.expanded_queries,
        )

        return {
            "trace_id": response.trace_id,
            "query_analysis": result.model_dump(),
            "rewritten_question": result.rewritten_query,
            "expanded_queries": result.expanded_queries,
            "retrieval_queries": retrieval_queries,
            "retrieval_mode": result.retrieval_mode,
            "target_doc_types": result.target_doc_types,
            "business_domain": result.business_domain or business_domain,
            "need_clarification": result.need_clarification,
            "clarification_question": result.clarification_question,
            "current_node": "analyze_query",
            "status": "QUERY_ANALYZED",
        }

    def route_after_query_analysis(self, state: RagState) -> str:
        """
        条件边：根据 retrieval_mode 选择检索节点。

        LangGraph 的 conditional edge 会调用这个函数，
        返回的字符串会映射到具体节点。
        """

        mode = (state.get("retrieval_mode") or "hybrid").lower()

        if mode == "bm25":
            return "bm25"

        if mode == "vector":
            return "vector"

        return "hybrid"

    async def retrieve_bm25(self, state: RagState) -> dict:
        """
        节点 2A：BM25 检索。

        只依赖 ES，不依赖 Milvus / DashScope。
        """

        retriever = self.bm25_retriever_factory(
            top_k=state.get("top_k", 10),
            doc_type=self._single_doc_type_or_none(state),
            business_domain=state.get("business_domain"),
        )

        documents = []

        for query in state.get("retrieval_queries", [state["question"]]):
            docs = await retriever.ainvoke(query)
            documents.extend(docs)

        return {
            "retrieved_documents": self.document_codec.documents_to_dicts(documents),
            "current_node": "retrieve_bm25",
            "status": "RETRIEVED",
        }

    async def retrieve_vector(self, state: RagState) -> dict:
        """
        节点 2B：向量检索。

        依赖 DashScope + Milvus。
        """

        retriever = self.vector_retriever_factory(
            top_k=state.get("top_k", 10),
            doc_type=self._single_doc_type_or_none(state),
            business_domain=state.get("business_domain"),
        )

        documents = []

        for query in state.get("retrieval_queries", [state["question"]]):
            docs = await retriever.ainvoke(query)
            documents.extend(docs)

        return {
            "retrieved_documents": self.document_codec.documents_to_dicts(documents),
            "current_node": "retrieve_vector",
            "status": "RETRIEVED",
        }

    async def retrieve_hybrid(self, state: RagState) -> dict:
        """
        节点 2C：Hybrid 检索。

        同时依赖 ES + Milvus + DashScope。
        """

        retriever = self.hybrid_retriever_factory(
            top_k=state.get("top_k", 10),
            vector_top_k=state.get("top_k", 10),
            bm25_top_k=state.get("top_k", 10),
            doc_type=self._single_doc_type_or_none(state),
            business_domain=state.get("business_domain"),
        )

        documents = []

        for query in state.get("retrieval_queries", [state["question"]]):
            docs = await retriever.ainvoke(query)
            documents.extend(docs)

        return {
            "retrieved_documents": self.document_codec.documents_to_dicts(documents),
            "current_node": "retrieve_hybrid",
            "status": "RETRIEVED",
        }

    def merge_documents(self, state: RagState) -> dict:
        """
        节点 3：合并文档。

        为什么要合并？
        因为 multi-query 或 Hybrid 可能召回同一个 chunk 多次。
        这里按 chunk_id 去重，保留最高分。
        """

        documents = state.get("retrieved_documents", [])

        merged_by_chunk_id: dict[int, dict] = {}

        for document in documents:
            chunk_id = document.get("chunk_id")

            if chunk_id is None:
                continue

            current_score = float(document.get("score") or 0.0)

            if chunk_id not in merged_by_chunk_id:
                merged_by_chunk_id[chunk_id] = document
                continue

            old_score = float(merged_by_chunk_id[chunk_id].get("score") or 0.0)

            if current_score > old_score:
                merged_by_chunk_id[chunk_id] = document

        merged_documents = sorted(
            merged_by_chunk_id.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

        return {
            "merged_documents": merged_documents,
            "current_node": "merge_documents",
            "status": "DOCUMENTS_MERGED",
        }

    def judge_retrieval_quality(self, state: RagState) -> dict:
        """
        节点 4：召回质量判断。

        当前只做基础判断：
        - 有没有结果；
        - 结果数量是否太少。
        """

        merged_documents = state.get("merged_documents", [])
        quality = self.quality_judge.judge(merged_documents)

        return {
            "retrieval_quality": quality,
            "need_rewrite": quality["need_rewrite"],
            "current_node": "judge_retrieval_quality",
            "status": "QUALITY_JUDGED",
        }

    def finish(self, state: RagState) -> dict:
        """
        结束工作流。

        模块 11 后：
        - 没有召回：RETRIEVAL_INSUFFICIENT
        - 有 context：CONTEXT_READY
        - 有 rerank 但 context 为空：RERANK_READY
        """

        quality = state.get("retrieval_quality", {})

        if quality.get("quality") == "POOR":
            return {
                "current_node": "finish",
                "status": "RETRIEVAL_INSUFFICIENT",
            }

        if state.get("context"):
            return {
                "current_node": "finish",
                "status": "CONTEXT_READY",
            }

        if state.get("reranked_documents"):
            return {
                "current_node": "finish",
                "status": "RERANK_READY",
            }

        return {
            "current_node": "finish",
            "status": "RETRIEVAL_READY",
        }

    def _build_retrieval_queries(
        self,
        rewritten_query: str,
        expanded_queries: list[str],
    ) -> list[str]:
        """
        生成最终检索 query 列表。

        规则：
        1. rewritten_query 放第一位；
        2. expanded_queries 追加；
        3. 去重；
        4. 最多保留 5 条，避免召回噪声太大。
        """

        queries = [rewritten_query]
        queries.extend(expanded_queries or [])

        return list(dict.fromkeys([query for query in queries if query]))[:5]

    def _single_doc_type_or_none(self, state: RagState) -> str | None:
        """
        如果只命中一个目标文档类型，就作为 Retriever filter。

        如果有多个目标文档类型，就不强行过滤。
        因为多个类型时直接过滤一个会丢召回。
        """

        doc_types = state.get("target_doc_types") or []

        if len(doc_types) == 1:
            return doc_types[0]

        return None
