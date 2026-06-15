from langgraph.graph import END, START, StateGraph

from src.rag_platform.application.query_understanding_service import QueryUnderstandingService
from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.rag_state import RagState
from src.rag_platform.rag.adaptive.models import (
    RetrievalQualityLevel,
    RetryStrategy,
)
from src.rag_platform.rag.adaptive.intermediate_fact_extractor import (
    IntermediateFactExtractor,
)
from src.rag_platform.rag.adaptive.multi_round_fusion import MultiRoundFusion
from src.rag_platform.rag.adaptive.quality_features import (
    extract_exact_terms,
    extract_retrieval_quality_features,
    has_comparison_intent,
)
from src.rag_platform.rag.adaptive.quality_policy import (
    RetrievalQualityPolicy,
)
from src.rag_platform.rag.adaptive.query_decomposer import QueryDecomposer
from src.rag_platform.rag.adaptive.query_rewriter import QueryRewriter
from src.rag_platform.rag.adaptive.sub_query_fusion import SubQueryFusion
from src.rag_platform.rag.graph.document_codec import GraphDocumentCodec
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
        settings=None,
        quality_policy=None,
        query_rewriter=None,
        multi_round_fusion=None,
        query_decomposer=None,
        sub_query_fusion=None,
        intermediate_fact_extractor=None,
    ) -> None:
        self.settings = settings or get_settings()
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
        self.quality_policy = (
            quality_policy
            or RetrievalQualityPolicy(self.settings)
        )
        self.query_rewriter = (
            query_rewriter
            or QueryRewriter(settings=self.settings)
        )
        self.multi_round_fusion = (
            multi_round_fusion
            or MultiRoundFusion(
                rank_constant=self.settings.rrf_rank_constant
            )
        )
        self.query_decomposer = (
            query_decomposer
            or QueryDecomposer(settings=self.settings)
        )
        self.sub_query_fusion = (
            sub_query_fusion
            or SubQueryFusion(
                rank_constant=self.settings.rrf_rank_constant,
                min_candidates=int(
                    getattr(
                        self.settings,
                        "sub_query_min_candidates",
                        1,
                    )
                ),
            )
        )
        self.intermediate_fact_extractor = (
            intermediate_fact_extractor
            or IntermediateFactExtractor(settings=self.settings)
        )

    def build(self):
        """
        构建并编译 LangGraph。

        StateGraph 是 builder，添加节点和边之后，需要 compile() 成可执行 graph。
        """

        graph = StateGraph(RagState)

        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("decompose_query", self.decompose_query)
        graph.add_node("retrieve_bm25", self.retrieve_bm25)
        graph.add_node("retrieve_vector", self.retrieve_vector)
        graph.add_node("retrieve_hybrid", self.retrieve_hybrid)
        graph.add_node("merge_documents", self.merge_documents)
        graph.add_node("rerank_documents", self.rerank_documents)
        graph.add_node("judge_retrieval_quality", self.judge_retrieval_quality)
        graph.add_node(
            "prepare_dependent_hop",
            self.prepare_dependent_hop,
        )
        graph.add_node("prepare_retry", self.prepare_retry)
        graph.add_node("build_context", self.build_context)
        graph.add_node("finish", self.finish)

        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "decompose_query")

        graph.add_conditional_edges(
            "decompose_query",
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

        graph.add_edge("merge_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "judge_retrieval_quality")
        graph.add_conditional_edges(
            "judge_retrieval_quality",
            self.route_after_quality_judge,
            {
                "context": "build_context",
                "dependent": "prepare_dependent_hop",
                "retry": "prepare_retry",
                "finish": "finish",
            },
        )
        graph.add_conditional_edges(
            "prepare_dependent_hop",
            self.route_after_query_analysis,
            {
                "bm25": "retrieve_bm25",
                "vector": "retrieve_vector",
                "hybrid": "retrieve_hybrid",
            },
        )
        graph.add_conditional_edges(
            "prepare_retry",
            self.route_after_query_analysis,
            {
                "bm25": "retrieve_bm25",
                "vector": "retrieve_vector",
                "hybrid": "retrieve_hybrid",
            },
        )
        graph.add_edge("build_context", "finish")
        graph.add_edge("finish", END)

        return graph.compile()

    async def decompose_query(self, state: RagState) -> dict:
        """
        节点：将复杂问题拆成最多三个可独立检索的子问题。

        分解失败时保留 Query 理解阶段生成的检索词，避免让新增能力
        阻断原有简单查询和 M9 自适应检索链路。
        """

        current_queries = list(
            state.get("retrieval_queries")
            or [state.get("question") or ""]
        )
        try:
            result = await self.query_decomposer.decompose(
                question=state.get("question") or "",
                rewritten_question=state.get("rewritten_question"),
                target_doc_types=list(
                    state.get("target_doc_types") or []
                ),
                need_clarification=bool(
                    state.get("need_clarification")
                ),
            )
        except Exception as exc:
            return {
                "decomposition": {
                    "requires_decomposition": False,
                    "sub_queries": [],
                    "reason": (
                        "查询分解执行异常，回退原检索词；"
                        f"{type(exc).__name__}"
                    ),
                    "fallback_used": True,
                },
                "retrieval_tasks": [],
                "retrieval_queries": current_queries,
                "current_node": "decompose_query",
                "status": "QUERY_DECOMPOSITION_FALLBACK",
            }

        decomposition = result.to_dict()
        all_tasks = [item.to_dict() for item in result.sub_queries]
        is_dependent = (
            result.requires_decomposition
            and result.decomposition_type == "DEPENDENT"
        )
        retrieval_tasks = (
            all_tasks[:1] if is_dependent else all_tasks
        )
        retrieval_queries = (
            [item["question"] for item in retrieval_tasks]
            if result.requires_decomposition
            else current_queries
        )
        updates = {
            "decomposition": decomposition,
            "retrieval_tasks": retrieval_tasks,
            "retrieval_queries": retrieval_queries,
            "anchor_retrieval_queries": (
                current_queries
                if result.requires_decomposition
                else []
            ),
            "current_node": "decompose_query",
            "status": (
                "QUERY_DECOMPOSED"
                if result.requires_decomposition
                else "QUERY_DECOMPOSITION_SKIPPED"
            ),
        }
        if is_dependent:
            updates.update(
                {
                    "max_retrieval_rounds": max(
                        int(
                            state.get("max_retrieval_rounds")
                            or 1
                        ),
                        int(
                            getattr(
                                self.settings,
                                "dependent_multi_hop_max_hops",
                                2,
                            )
                        ),
                    ),
                    "dependent_hop": {
                        "enabled": True,
                        "status": "FIRST_HOP_READY",
                        "current_hop": 1,
                        "max_hops": int(
                            getattr(
                                self.settings,
                                "dependent_multi_hop_max_hops",
                                2,
                            )
                        ),
                        "first_hop_question": (
                            retrieval_tasks[0]["question"]
                        ),
                        "next_query_template": (
                            all_tasks[1]["question"]
                        ),
                        "intermediate_fact": "",
                        "evidence_quote": "",
                        "supporting_chunk_id": None,
                        "second_hop_query": "",
                        "confidence": 0.0,
                        "fallback_used": False,
                    },
                }
            )
        return updates

    def build_context(self, state: RagState) -> dict:
        """
        节点：Context 构建。

        优先使用 reranked_documents。
        如果 reranked_documents 为空，则 fallback 到 merged_documents。
        """

        trace_id = state.get("trace_id") or ""
        query = (
            state.get("final_retrieval_query")
            or state.get("rewritten_question")
            or state.get("question")
            or ""
        )

        documents = state.get("reranked_documents") or state.get("merged_documents") or []

        retrieval_tasks = list(
            (state.get("decomposition") or {}).get(
                "sub_queries",
                [],
            )
        )
        build_kwargs = {
            "trace_id": trace_id,
            "query_text": query,
            "documents": documents,
        }
        if (
            (state.get("decomposition") or {}).get(
                "requires_decomposition"
            )
            and retrieval_tasks
        ):
            build_kwargs["sub_queries"] = retrieval_tasks
        result, context_build_info = (
            self.context_build_service.build_context(**build_kwargs)
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

        质量不足且还有轮次时重试。
        达到最大轮次后，有候选则继续构建Context，无候选则结束。
        """

        quality = state.get("retrieval_quality", {})
        quality_level = quality.get("quality")
        retry_strategy = quality.get("retry_strategy", "NONE")
        current_round = int(state.get("retrieval_round") or 1)
        max_rounds = int(
            state.get("max_retrieval_rounds")
            or self.settings.adaptive_max_rounds
        )
        if self._dependent_hop_pending(state):
            return "dependent"
        if (
            self.settings.adaptive_retrieval_enabled
            and retry_strategy != RetryStrategy.NONE.value
            and current_round < max_rounds
        ):
            return "retry"
        if (
            state.get("reranked_documents")
            or state.get("merged_documents")
        ):
            return "context"
        if quality_level == "POOR":
            return "finish"
        return "finish"

    async def prepare_dependent_hop(
        self,
        state: RagState,
    ) -> dict:
        decomposition = dict(state.get("decomposition") or {})
        sub_queries = [
            dict(item)
            for item in decomposition.get("sub_queries") or []
        ]
        dependent_hop = dict(state.get("dependent_hop") or {})
        if len(sub_queries) < 2:
            return {
                "dependent_hop": {
                    **dependent_hop,
                    "status": "INVALID_PLAN",
                    "fallback_used": True,
                    "reason": "顺序依赖计划缺少第二跳",
                },
                "current_node": "prepare_dependent_hop",
                "status": "DEPENDENT_HOP_INVALID",
            }

        first_task = sub_queries[0]
        second_task = sub_queries[1]
        next_query_template = str(
            second_task.get("question") or ""
        )
        candidates = list(
            state.get("reranked_documents")
            or state.get("merged_documents")
            or []
        )
        try:
            fact_result = await self.intermediate_fact_extractor.extract(
                question=state.get("question") or "",
                first_hop_question=str(
                    first_task.get("question") or ""
                ),
                next_query_template=next_query_template,
                candidate_documents=candidates,
            )
        except Exception as exc:
            fact_result = None
            extraction_reason = (
                "中间事实抽取异常；"
                f"{type(exc).__name__}"
            )
        else:
            extraction_reason = fact_result.reason

        if fact_result is not None and fact_result.success:
            second_hop_query = self._resolve_dependent_query(
                next_query_template,
                fact_result.intermediate_fact,
            )
            hop_status = "SECOND_HOP_READY"
            fallback_used = False
            intermediate_fact = fact_result.intermediate_fact
            evidence_quote = fact_result.evidence_quote
            supporting_chunk_id = (
                fact_result.supporting_chunk_id
            )
            confidence = fact_result.confidence
        else:
            second_hop_query = state.get("question") or ""
            hop_status = "SECOND_HOP_FALLBACK"
            fallback_used = True
            intermediate_fact = ""
            evidence_quote = ""
            supporting_chunk_id = None
            confidence = 0.0

        resolved_second_task = {
            **second_task,
            "question": second_hop_query,
            "is_template": False,
            "resolved_from_intermediate_fact": bool(
                intermediate_fact
            ),
        }
        decomposition["sub_queries"] = [
            first_task,
            resolved_second_task,
        ]
        return {
            "decomposition": decomposition,
            "dependent_hop": {
                **dependent_hop,
                "enabled": True,
                "status": hop_status,
                "current_hop": 2,
                "intermediate_fact": intermediate_fact,
                "evidence_quote": evidence_quote,
                "supporting_chunk_id": supporting_chunk_id,
                "second_hop_query": second_hop_query,
                "confidence": confidence,
                "reason": extraction_reason,
                "fallback_used": fallback_used,
            },
            "retrieval_round": int(
                state.get("retrieval_round") or 1
            ) + 1,
            "retry_strategy": "DEPENDENT_HOP",
            "query_variant": "DEPENDENT_HOP",
            "retrieval_queries": [second_hop_query],
            "anchor_retrieval_queries": [],
            "retrieval_tasks": [resolved_second_task],
            "sub_query_coverage": {
                "total_sub_queries": 0,
                "covered_sub_queries": 0,
                "coverage_rate": 1.0,
                "items": {},
            },
            "retrieved_documents": [],
            "reranked_documents": [],
            "current_node": "prepare_dependent_hop",
            "status": hop_status,
        }

    def _dependent_hop_pending(self, state: RagState) -> bool:
        decomposition = state.get("decomposition") or {}
        dependent_hop = state.get("dependent_hop") or {}
        if not bool(
            getattr(
                self.settings,
                "dependent_multi_hop_enabled",
                True,
            )
        ):
            return False
        if (
            not decomposition.get("requires_decomposition")
            or decomposition.get("decomposition_type")
            != "DEPENDENT"
        ):
            return False
        if dependent_hop.get("status") != "FIRST_HOP_READY":
            return False
        if int(state.get("retrieval_round") or 1) >= int(
            state.get("max_retrieval_rounds") or 2
        ):
            return False
        return bool(
            state.get("reranked_documents")
            or state.get("merged_documents")
        )

    @staticmethod
    def _resolve_dependent_query(
        template: str,
        intermediate_fact: str,
    ) -> str:
        query = template.replace(
            "{{intermediate_fact}}",
            intermediate_fact,
        ).replace(
            "{intermediate_fact}",
            intermediate_fact,
        )
        if query == template:
            query = f"{intermediate_fact} {template}"
        return " ".join(query.split()).strip()

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
        decomposition = state.get("decomposition") or {}
        query = (
            state.get("question")
            if decomposition.get("requires_decomposition")
            else (
                (state.get("retrieval_queries") or [None])[0]
                or state.get("rewritten_question")
                or state.get("question")
            )
        ) or ""
        documents = state.get("merged_documents", [])
        retrieval_tasks = list(
            decomposition.get("sub_queries")
            or state.get("retrieval_tasks")
            or []
        )
        effective_top_n = int(
            getattr(
                self.settings,
                "rerank_top_n",
                len(documents) or 1,
            )
        )
        rerank_kwargs = {
            "trace_id": trace_id,
            "query": query,
            "documents": documents,
        }
        if (
            decomposition.get("requires_decomposition")
            and retrieval_tasks
        ):
            effective_top_n += min(
                len(retrieval_tasks),
                int(
                    getattr(
                        self.settings,
                        "query_decomposition_rerank_extra_limit",
                        3,
                    )
                ),
            )
            rerank_kwargs["top_n"] = effective_top_n

        reranked_documents, rerank_info = (
            await self.rerank_service.rerank_documents(
                **rerank_kwargs
            )
        )

        coverage = {
            "total_sub_queries": 0,
            "covered_sub_queries": 0,
            "coverage_rate": 1.0,
            "items": {},
        }
        if (
            decomposition.get("requires_decomposition")
            and retrieval_tasks
        ):
            sub_query_ids = [
                str(item["sub_query_id"])
                for item in retrieval_tasks
            ]
            reranked_documents = (
                self.sub_query_fusion.restore_rerank_quota(
                    reranked_documents=reranked_documents,
                    candidate_documents=documents,
                    sub_query_ids=sub_query_ids,
                    top_n=effective_top_n,
                    quota=int(
                        getattr(
                            self.settings,
                            "sub_query_rerank_quota",
                            1,
                        )
                    ),
                )
            )
            coverage = self.sub_query_fusion.calculate_coverage(
                sub_queries=retrieval_tasks,
                candidate_documents=documents,
                final_documents=reranked_documents,
            )

        return {
            "reranked_documents": reranked_documents,
            "rerank_info": rerank_info,
            "sub_query_coverage": coverage,
            "final_retrieval_query": query,
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
            "anchor_retrieval_queries": [],
            "retrieval_mode": result.retrieval_mode,
            "target_doc_types": result.target_doc_types,
            "decomposition": {
                "requires_decomposition": False,
                "sub_queries": [],
                "reason": "尚未执行查询分解",
                "fallback_used": False,
            },
            "retrieval_tasks": [],
            "sub_query_coverage": {
                "total_sub_queries": 0,
                "covered_sub_queries": 0,
                "coverage_rate": 1.0,
                "items": {},
            },
            "dependent_hop": {
                "enabled": False,
                "status": "NOT_REQUIRED",
                "current_hop": 0,
                "max_hops": int(
                    getattr(
                        self.settings,
                        "dependent_multi_hop_max_hops",
                        2,
                    )
                ),
                "intermediate_fact": "",
                "evidence_quote": "",
                "supporting_chunk_id": None,
                "second_hop_query": "",
                "confidence": 0.0,
                "fallback_used": False,
            },
            "business_domain": result.business_domain or business_domain,
            "initial_business_domain": result.business_domain or business_domain,
            "initial_target_doc_types": result.target_doc_types,
            "need_clarification": result.need_clarification,
            "clarification_question": result.clarification_question,
            "retrieval_round": 1,
            "max_retrieval_rounds": (
                self.settings.adaptive_max_rounds
                if self.settings.adaptive_retrieval_enabled
                else 1
            ),
            "retrieval_attempts": [],
            "retry_strategy": "INITIAL",
            "query_variant": "ORIGINAL",
            "removed_filters": [],
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

        documents = []
        for task in self._retrieval_task_specs(state):
            retriever = self.bm25_retriever_factory(
                top_k=state.get("top_k", 10),
                doc_type=self._task_doc_type_or_default(
                    task,
                    state,
                ),
                business_domain=state.get("business_domain"),
            )
            documents.extend(
                await self._retrieve_queries(
                    retriever,
                    state,
                    retrieval_tasks=[task] if task else None,
                )
            )
        anchor_queries = self._anchor_retrieval_queries(state)
        if anchor_queries:
            retriever = self.bm25_retriever_factory(
                top_k=state.get("top_k", 10),
                doc_type=self._single_doc_type_or_none(state),
                business_domain=state.get("business_domain"),
            )
            documents.extend(
                await self._retrieve_queries(
                    retriever,
                    state,
                    queries_override=anchor_queries,
                    anchor_query=True,
                )
            )

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

        documents = []
        for task in self._retrieval_task_specs(state):
            retriever = self.vector_retriever_factory(
                top_k=state.get("top_k", 10),
                doc_type=self._task_doc_type_or_default(
                    task,
                    state,
                ),
                business_domain=state.get("business_domain"),
            )
            documents.extend(
                await self._retrieve_queries(
                    retriever,
                    state,
                    retrieval_tasks=[task] if task else None,
                )
            )
        anchor_queries = self._anchor_retrieval_queries(state)
        if anchor_queries:
            retriever = self.vector_retriever_factory(
                top_k=state.get("top_k", 10),
                doc_type=self._single_doc_type_or_none(state),
                business_domain=state.get("business_domain"),
            )
            documents.extend(
                await self._retrieve_queries(
                    retriever,
                    state,
                    queries_override=anchor_queries,
                    anchor_query=True,
                )
            )

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

        documents = []
        for task in self._retrieval_task_specs(state):
            retriever = self.hybrid_retriever_factory(
                top_k=state.get("top_k", 10),
                vector_top_k=state.get("top_k", 10),
                bm25_top_k=state.get("top_k", 10),
                doc_type=self._task_doc_type_or_default(
                    task,
                    state,
                ),
                business_domain=state.get("business_domain"),
            )
            documents.extend(
                await self._retrieve_queries(
                    retriever,
                    state,
                    retrieval_tasks=[task] if task else None,
                )
            )
        anchor_queries = self._anchor_retrieval_queries(state)
        if anchor_queries:
            retriever = self.hybrid_retriever_factory(
                top_k=state.get("top_k", 10),
                vector_top_k=state.get("top_k", 10),
                bm25_top_k=state.get("top_k", 10),
                doc_type=self._single_doc_type_or_none(state),
                business_domain=state.get("business_domain"),
            )
            documents.extend(
                await self._retrieve_queries(
                    retriever,
                    state,
                    queries_override=anchor_queries,
                    anchor_query=True,
                )
            )

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
        candidate_limit = max(
            int(state.get("top_k") or 10),
            int(
                getattr(
                    self.settings,
                    "rerank_candidate_limit",
                    state.get("top_k") or 10,
                )
            ),
        )
        decomposition = state.get("decomposition") or {}
        retrieval_tasks = list(state.get("retrieval_tasks") or [])
        if (
            decomposition.get("requires_decomposition")
            and retrieval_tasks
        ):
            task_results = []
            for task in retrieval_tasks:
                sub_query_id = str(task["sub_query_id"])
                task_results.append(
                    {
                        **task,
                        "documents": [
                            document
                            for document in documents
                            if sub_query_id
                            in (
                                document.get("metadata") or {}
                            ).get("sub_query_ids", [])
                        ],
                    }
                )
            anchor_documents = [
                document
                for document in documents
                if (document.get("metadata") or {}).get(
                    "anchor_query"
                )
            ]
            if anchor_documents:
                task_results.append(
                    {
                        "sub_query_id": "ANCHOR",
                        "question": (
                            state.get("question") or ""
                        ),
                        "documents": anchor_documents,
                    }
                )
            round_documents = self.sub_query_fusion.fuse(
                task_results,
                top_k=candidate_limit,
            )
        else:
            merged_by_chunk_id: dict[int, dict] = {}
            for document in documents:
                chunk_id = document.get("chunk_id")
                if chunk_id is None:
                    continue
                current_score = float(
                    document.get("score") or 0.0
                )
                previous = merged_by_chunk_id.get(chunk_id)
                if (
                    previous is None
                    or current_score
                    > float(previous.get("score") or 0.0)
                ):
                    merged_by_chunk_id[chunk_id] = document
            round_documents = sorted(
                merged_by_chunk_id.values(),
                key=lambda item: float(item.get("score") or 0.0),
                reverse=True,
            )
        attempt = {
            "round_no": int(state.get("retrieval_round") or 1),
            "strategy": state.get("retry_strategy") or "INITIAL",
            "query_variant": state.get("query_variant") or "ORIGINAL",
            "queries": list(
                state.get("retrieval_queries")
                or [state.get("question") or ""]
            ),
            "retrieval_mode": (
                state.get("retrieval_mode") or "hybrid"
            ),
            "doc_type_filter": self._single_doc_type_or_none(state),
            "business_domain_filter": state.get("business_domain"),
            "removed_filters": list(
                state.get("removed_filters") or []
            ),
            "dependent_hop": dict(
                state.get("dependent_hop") or {}
            ),
            "documents": round_documents,
        }
        attempts = [
            *list(state.get("retrieval_attempts") or []),
            attempt,
        ]
        merged_documents = self.multi_round_fusion.fuse(
            attempts,
            top_k=candidate_limit,
        )

        return {
            "merged_documents": merged_documents,
            "retrieval_attempts": attempts,
            "current_node": "merge_documents",
            "status": "DOCUMENTS_MERGED",
        }

    def judge_retrieval_quality(self, state: RagState) -> dict:
        """
        节点 4：召回质量判断。

        计算确定性质量特征，再由可配置策略作出决策。
        """

        merged_documents = state.get("merged_documents", [])
        reranked_documents = state.get("reranked_documents", [])
        features = extract_retrieval_quality_features(
            question=state.get("question") or "",
            documents=merged_documents,
            reranked_documents=reranked_documents,
            target_doc_types=state.get("target_doc_types") or [],
        )
        decision = self.quality_policy.decide(features)
        quality = decision.to_dict()
        quality["features"] = features.to_dict()
        coverage = state.get("sub_query_coverage") or {}
        if (
            (state.get("decomposition") or {}).get(
                "requires_decomposition"
            )
            and float(coverage.get("coverage_rate") or 0.0) < 1.0
            and quality["retry_strategy"]
            == RetryStrategy.NONE.value
        ):
            reasons = [
                *list(quality.get("reasons") or []),
                "子问题证据覆盖不足",
            ]
            quality.update(
                {
                    "quality": RetrievalQualityLevel.WEAK.value,
                    "retry_strategy": (
                        RetryStrategy.QUERY_REWRITE.value
                    ),
                    "need_rewrite": True,
                    "reasons": reasons,
                    "reason": "；".join(reasons),
                }
            )
        attempts = list(state.get("retrieval_attempts") or [])
        if attempts:
            attempts[-1] = {
                **attempts[-1],
                "reranked_documents": reranked_documents,
                "quality": quality,
                "sub_query_coverage": coverage,
            }

        updates = {
            "retrieval_quality": quality,
            "quality_features": features.to_dict(),
            "need_rewrite": quality["need_rewrite"],
            "retrieval_attempts": attempts,
            "current_node": "judge_retrieval_quality",
            "status": "QUALITY_JUDGED",
        }
        decomposition = state.get("decomposition") or {}
        if (
            decomposition.get("decomposition_type") == "DEPENDENT"
            and int(state.get("retrieval_round") or 1) >= 2
        ):
            updates["dependent_hop"] = {
                **dict(state.get("dependent_hop") or {}),
                "status": "COMPLETED",
            }
        return updates

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

    async def prepare_retry(self, state: RagState) -> dict:
        quality = state.get("retrieval_quality") or {}
        strategy = RetryStrategy(
            quality.get("retry_strategy")
            or RetryStrategy.QUERY_REWRITE.value
        )
        reasons = list(quality.get("reasons") or [])
        current_queries = list(
            state.get("retrieval_queries")
            or [state.get("question") or ""]
        )
        retrieval_mode = state.get("retrieval_mode") or "hybrid"
        target_doc_types = list(
            state.get("target_doc_types") or []
        )
        business_domain = state.get("business_domain")
        removed_filters: list[str] = []
        query_variant = "REWRITTEN"
        retrieval_tasks: list[dict] = []

        if strategy == RetryStrategy.FORCE_BM25:
            exact_terms = extract_exact_terms(
                state.get("question") or ""
            )
            exact_query = " ".join(
                [*exact_terms, state.get("question") or ""]
            ).strip()
            retrieval_queries = self._deduplicate_queries(
                [exact_query, *current_queries]
            )
            retrieval_mode = "bm25"
            query_variant = "EXACT"
            if len(target_doc_types) == 1:
                target_doc_types = []
                removed_filters.append("doc_type")
        elif strategy == RetryStrategy.RELAX_FILTER:
            retrieval_queries = current_queries
            query_variant = "RELAXED"
            if target_doc_types:
                target_doc_types = []
                removed_filters.append("doc_type")
            elif business_domain:
                business_domain = None
                removed_filters.append("business_domain")
        else:
            coverage_items = (
                state.get("sub_query_coverage") or {}
            ).get("items") or {}
            uncovered_tasks = [
                task
                for task in (state.get("retrieval_tasks") or [])
                if not (
                    coverage_items.get(
                        str(task.get("sub_query_id") or ""),
                        {},
                    ).get("covered")
                )
            ]
            if (
                (state.get("decomposition") or {}).get(
                    "requires_decomposition"
                )
                and uncovered_tasks
                and "子问题证据覆盖不足" in reasons
            ):
                for task in uncovered_tasks:
                    sub_query_id = str(task["sub_query_id"])
                    rewrite_result = await self.query_rewriter.rewrite(
                        original_question=task["question"],
                        current_queries=[task["question"]],
                        quality_reasons=reasons,
                        candidate_documents=[
                            document
                            for document in state.get(
                                "merged_documents",
                                [],
                            )
                            if sub_query_id
                            in (
                                document.get("metadata") or {}
                            ).get("sub_query_ids", [])
                        ],
                    )
                    retrieval_tasks.append(
                        {
                            **task,
                            "question": (
                                rewrite_result.rewritten_query
                                or task["question"]
                            ),
                        }
                    )
                retrieval_queries = [
                    task["question"] for task in retrieval_tasks
                ]
                query_variant = "SUB_QUERY_REWRITTEN"
            else:
                rewrite_result = await self.query_rewriter.rewrite(
                    original_question=state.get("question") or "",
                    current_queries=current_queries,
                    quality_reasons=reasons,
                    candidate_documents=state.get(
                        "merged_documents",
                        [],
                    ),
                )
                if has_comparison_intent(
                    state.get("question") or ""
                ):
                    question = state.get("question") or ""
                    retrieval_queries = self._deduplicate_queries(
                        [
                            rewrite_result.rewritten_query,
                            f"{question} V1 旧版 原规则",
                            f"{question} V2 新版 当前规则",
                            *rewrite_result.expanded_queries,
                        ]
                    )[:5]
                else:
                    retrieval_queries = rewrite_result.all_queries
                query_variant = (
                    "FALLBACK_REWRITE"
                    if rewrite_result.fallback_used
                    else "REWRITTEN"
                )
                if len(target_doc_types) == 1:
                    target_doc_types = []
                    removed_filters.append("doc_type")

        return {
            "retrieval_round": int(
                state.get("retrieval_round") or 1
            ) + 1,
            "retry_strategy": strategy.value,
            "query_variant": query_variant,
            "retrieval_queries": retrieval_queries,
            "anchor_retrieval_queries": [],
            "retrieval_mode": retrieval_mode,
            "target_doc_types": target_doc_types,
            "business_domain": business_domain,
            "removed_filters": removed_filters,
            "retrieval_tasks": retrieval_tasks,
            "sub_query_coverage": {
                "total_sub_queries": 0,
                "covered_sub_queries": 0,
                "coverage_rate": 1.0,
                "items": {},
            },
            "retrieved_documents": [],
            "reranked_documents": [],
            "current_node": "prepare_retry",
            "status": "RETRY_PREPARED",
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

    def _retrieval_task_specs(
        self,
        state: RagState,
    ) -> list[dict | None]:
        decomposition = state.get("decomposition") or {}
        retrieval_tasks = list(state.get("retrieval_tasks") or [])
        if (
            decomposition.get("requires_decomposition")
            and retrieval_tasks
        ):
            return retrieval_tasks
        return [None]

    def _task_doc_type_or_default(
        self,
        task: dict | None,
        state: RagState,
    ) -> str | None:
        if task is None:
            return self._single_doc_type_or_none(state)
        return None

    def _anchor_retrieval_queries(
        self,
        state: RagState,
    ) -> list[str]:
        if not (state.get("decomposition") or {}).get(
            "requires_decomposition"
        ):
            return []
        return list(state.get("anchor_retrieval_queries") or [])

    async def _retrieve_queries(
        self,
        retriever,
        state: RagState,
        retrieval_tasks: list[dict] | None = None,
        queries_override: list[str] | None = None,
        anchor_query: bool = False,
    ) -> list:
        documents = []
        decomposition = state.get("decomposition") or {}
        active_tasks = (
            retrieval_tasks
            if retrieval_tasks is not None
            else list(state.get("retrieval_tasks") or [])
        )
        if queries_override is not None:
            queries = [(query, "") for query in queries_override]
        elif (
            decomposition.get("requires_decomposition")
            and active_tasks
        ):
            queries = [
                (
                    str(task.get("question") or ""),
                    str(task.get("sub_query_id") or ""),
                )
                for task in active_tasks
            ]
        else:
            queries = [
                (query, "")
                for query in (
                    state.get("retrieval_queries")
                    or [state.get("question") or ""]
                )
            ]
        for query, sub_query_id in queries:
            docs = await retriever.ainvoke(query)
            for document in docs:
                metadata = dict(document.metadata or {})
                metadata.update(
                    {
                        "retrieval_query": query,
                        "retrieval_round": int(
                            state.get("retrieval_round") or 1
                        ),
                        "query_variant": (
                            state.get("query_variant")
                            or "ORIGINAL"
                        ),
                    }
                )
                dependent_hop = state.get("dependent_hop") or {}
                if dependent_hop.get("enabled"):
                    metadata["dependent_hop"] = int(
                        dependent_hop.get("current_hop") or 1
                    )
                if sub_query_id:
                    metadata["sub_query_ids"] = [sub_query_id]
                    metadata["sub_query_texts"] = [query]
                if anchor_query:
                    metadata["anchor_query"] = True
                document.metadata = metadata
            documents.extend(docs)
        return documents

    @staticmethod
    def _deduplicate_queries(
        queries: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                query.strip()
                for query in queries
                if query and query.strip()
            )
        )[:3]
