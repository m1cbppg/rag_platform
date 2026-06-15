from types import SimpleNamespace

from langchain_core.documents import Document

from src.rag_platform.rag.adaptive.models import (
    DecomposedSubQuery,
    IntermediateFactResult,
    QueryDecompositionResult,
    QueryRewriteResult,
    RetrievalQualityDecision,
    RetrievalQualityLevel,
    RetryStrategy,
)
from src.rag_platform.domain.context import (
    Citation,
    ContextBuildResult,
    ContextBuildStatus,
)
from src.rag_platform.rag.graph.rag_retrieval_graph import RagRetrievalGraphBuilder
from src.rag_platform.schemas.query_analysis import (
    QueryAnalysisResponse,
    QueryAnalysisResult,
)


class FakeQueryUnderstandingService:
    async def analyze(self, request) -> QueryAnalysisResponse:
        return QueryAnalysisResponse(
            trace_id="trace-from-query-service",
            result=QueryAnalysisResult(
                original_query=request.query,
                rewritten_query="退款规则",
                expanded_queries=["售后退款"],
                target_doc_types=["RULE"],
                retrieval_mode="bm25",
                business_domain=request.business_domain,
                confidence=0.9,
                reason="测试结果",
            ),
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def ainvoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return [
            Document(
                page_content=f"{query}的召回内容",
                metadata={
                    "chunk_id": len(self.queries),
                    "score": 0.9,
                    "title": "退款规则",
                },
            )
        ]


class SequencedRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def ainvoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        chunk_id = len(self.queries)
        return [
            Document(
                page_content=f"{query}的召回内容",
                metadata={
                    "chunk_id": chunk_id,
                    "doc_id": 100 + chunk_id,
                    "score": 0.9,
                    "title": "订单规则",
                    "chunk_type": "RULE",
                    "version": f"V{chunk_id}",
                    "source": "bm25",
                },
            )
        ]


class FakeRetrieverFactory:
    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.calls: list[dict] = []
        self.retriever = FakeRetriever()

    def __call__(self, **kwargs) -> FakeRetriever:
        self.kwargs = kwargs
        self.calls.append(kwargs)
        return self.retriever


class SequencedRetrieverFactory(FakeRetrieverFactory):
    def __init__(self) -> None:
        super().__init__()
        self.retriever = SequencedRetriever()


class FakeRerankService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def rerank_documents(
        self,
        trace_id,
        query,
        documents,
        top_n=None,
    ):
        self.calls.append(
            {
                "trace_id": trace_id,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
        )
        return documents, {"status": "SUCCESS", "trace_id": trace_id, "query": query}


class FakeQueryRewriter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def rewrite(self, **kwargs) -> QueryRewriteResult:
        self.calls.append(kwargs)
        return QueryRewriteResult(
            rewritten_query="订单取消规则 新版 旧版",
            expanded_queries=[],
            reason="补充版本检索词",
        )


class FakeQueryDecomposer:
    def __init__(
        self,
        result: QueryDecompositionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or QueryDecompositionResult(
            requires_decomposition=False,
            reason="简单问题",
        )
        self.error = error
        self.calls: list[dict] = []

    async def decompose(self, **kwargs) -> QueryDecompositionResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeIntermediateFactExtractor:
    def __init__(self, result: IntermediateFactResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def extract(self, **kwargs) -> IntermediateFactResult:
        self.calls.append(kwargs)
        return self.result


class SequencedQualityPolicy:
    def __init__(self, decisions) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, features):
        decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return decision


class FakeContextBuildService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def build_context(
        self,
        trace_id,
        query_text,
        documents,
        sub_queries=None,
    ):
        self.calls.append(
            {
                "trace_id": trace_id,
                "query_text": query_text,
                "documents": documents,
                "sub_queries": sub_queries,
            }
        )
        citation = Citation(
            citation_id="C1",
            chunk_id=1,
            doc_id=10,
            title="退款规则",
            title_path=None,
            source_section=None,
            chunk_type="RULE",
            expansion_type="SELF",
            sort_order=1,
        )
        return (
            ContextBuildResult(
                context="退款规则正文",
                citations=[citation],
                used_chunks=[],
                estimated_tokens=10,
                status=ContextBuildStatus.SUCCESS,
                message="ok",
            ),
            {"context_log_id": 1, "query": query_text, "trace_id": trace_id},
        )


async def test_graph_nodes_use_injected_services_and_retriever_factory() -> None:
    bm25_factory = FakeRetrieverFactory()
    builder = RagRetrievalGraphBuilder(
        query_understanding_service=FakeQueryUnderstandingService(),
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
        bm25_retriever_factory=bm25_factory,
    )
    initial_state = {
        "question": "怎么退款？",
        "session_id": "session-1",
        "business_domain": "after_sales",
        "trace_id": "initial-trace",
        "top_k": 5,
    }

    analysis = await builder.analyze_query(initial_state)
    retrieved = await builder.retrieve_bm25({**initial_state, **analysis})
    reranked = await builder.rerank_documents(
        {
            **initial_state,
            **analysis,
            "merged_documents": retrieved["retrieved_documents"],
        }
    )
    context = builder.build_context(
        {
            **initial_state,
            **analysis,
            **reranked,
        }
    )

    assert analysis["trace_id"] == "trace-from-query-service"
    assert analysis["retrieval_queries"] == ["退款规则", "售后退款"]
    assert bm25_factory.kwargs == {
        "top_k": 5,
        "doc_type": "RULE",
        "business_domain": "after_sales",
    }
    assert bm25_factory.retriever.queries == ["退款规则", "售后退款"]
    assert context["context"] == "退款规则正文"
    assert context["citations"][0]["citation_id"] == "C1"


async def test_compiled_graph_retries_weak_retrieval_once() -> None:
    bm25_factory = SequencedRetrieverFactory()
    query_rewriter = FakeQueryRewriter()
    quality_policy = SequencedQualityPolicy(
        [
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.WEAK,
                score=0.5,
                retry_strategy=RetryStrategy.QUERY_REWRITE,
                reasons=["版本证据不足"],
            ),
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.GOOD,
                score=0.8,
                retry_strategy=RetryStrategy.NONE,
                reasons=["证据充分"],
            ),
        ]
    )
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
        rerank_enabled=True,
    )
    graph = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
        bm25_retriever_factory=bm25_factory,
        query_rewriter=query_rewriter,
        quality_policy=quality_policy,
    ).build()

    result = await graph.ainvoke(
        {
            "question": "订单取消规则新旧版本有什么不同？",
            "session_id": "session-1",
            "business_domain": "after_sales",
            "trace_id": "trace-1",
            "top_k": 5,
            "status": "STARTED",
        }
    )

    assert result["status"] == "CONTEXT_READY"
    assert result["retrieval_round"] == 2
    assert len(result["retrieval_attempts"]) == 2
    assert result["retrieval_attempts"][0]["quality"]["quality"] == "WEAK"
    assert result["retrieval_attempts"][1]["quality"]["quality"] == "GOOD"
    assert result["retrieval_attempts"][1]["strategy"] == "QUERY_REWRITE"
    assert len(query_rewriter.calls) == 1
    assert "订单取消规则 新版 旧版" in bm25_factory.retriever.queries
    assert any(
        "V1 旧版" in query
        for query in bm25_factory.retriever.queries
    )
    assert any(
        "V2 新版" in query
        for query in bm25_factory.retriever.queries
    )
    assert bm25_factory.kwargs["doc_type"] is None
    assert quality_policy.calls == 2


async def test_compiled_graph_does_not_retry_good_retrieval() -> None:
    bm25_factory = SequencedRetrieverFactory()
    query_rewriter = FakeQueryRewriter()
    quality_policy = SequencedQualityPolicy(
        [
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.GOOD,
                score=0.9,
                retry_strategy=RetryStrategy.NONE,
                reasons=["证据充分"],
            )
        ]
    )
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
        rerank_enabled=True,
    )
    graph = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
        bm25_retriever_factory=bm25_factory,
        query_rewriter=query_rewriter,
        quality_policy=quality_policy,
    ).build()

    result = await graph.ainvoke(
        {
            "question": "怎么退款？",
            "session_id": "session-1",
            "business_domain": "after_sales",
            "trace_id": "trace-1",
            "top_k": 5,
            "status": "STARTED",
        }
    )

    assert result["retrieval_round"] == 1
    assert len(result["retrieval_attempts"]) == 1
    assert query_rewriter.calls == []


async def test_simple_query_decomposition_keeps_existing_retrieval_path() -> None:
    bm25_factory = SequencedRetrieverFactory()
    decomposer = FakeQueryDecomposer()
    rerank_service = FakeRerankService()
    quality_policy = SequencedQualityPolicy(
        [
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.GOOD,
                score=0.9,
                retry_strategy=RetryStrategy.NONE,
                reasons=["证据充分"],
            )
        ]
    )
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
        rerank_enabled=True,
        query_decomposition_enabled=True,
        sub_query_min_candidates=1,
        sub_query_rerank_quota=1,
        rerank_top_n=5,
    )
    graph = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        query_decomposer=decomposer,
        rerank_service=rerank_service,
        context_build_service=FakeContextBuildService(),
        bm25_retriever_factory=bm25_factory,
        quality_policy=quality_policy,
    ).build()

    result = await graph.ainvoke(
        {
            "question": "怎么退款？",
            "session_id": "session-1",
            "business_domain": "after_sales",
            "trace_id": "trace-1",
            "top_k": 5,
            "status": "STARTED",
        }
    )

    assert result["decomposition"]["requires_decomposition"] is False
    assert result["retrieval_tasks"] == []
    assert bm25_factory.retriever.queries == ["退款规则", "售后退款"]
    assert rerank_service.calls[0]["top_n"] is None


async def test_complex_query_retrieves_each_sub_query_and_preserves_quota() -> None:
    bm25_factory = SequencedRetrieverFactory()
    context_service = FakeContextBuildService()
    rerank_service = FakeRerankService()
    decomposer = FakeQueryDecomposer(
        QueryDecompositionResult(
            requires_decomposition=True,
            reason="包含两个信息需求",
            sub_queries=[
                DecomposedSubQuery(
                    sub_query_id="SQ1",
                    question="未出库订单修改地址条件",
                    target_doc_types=["FAQ"],
                ),
                DecomposedSubQuery(
                    sub_query_id="SQ2",
                    question="待审核售后单材料要求",
                    target_doc_types=["SOP"],
                ),
            ],
        )
    )
    quality_policy = SequencedQualityPolicy(
        [
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.GOOD,
                score=0.9,
                retry_strategy=RetryStrategy.NONE,
                reasons=["证据充分"],
            )
        ]
    )
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
        rerank_enabled=True,
        query_decomposition_enabled=True,
        sub_query_min_candidates=1,
        sub_query_rerank_quota=1,
        rerank_top_n=2,
        query_decomposition_rerank_extra_limit=3,
    )
    graph = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        query_decomposer=decomposer,
        rerank_service=rerank_service,
        context_build_service=context_service,
        bm25_retriever_factory=bm25_factory,
        quality_policy=quality_policy,
    ).build()

    result = await graph.ainvoke(
        {
            "question": "订单怎么改地址，同时售后单要补什么材料？",
            "session_id": "session-1",
            "business_domain": "after_sales",
            "trace_id": "trace-1",
            "top_k": 2,
            "status": "STARTED",
        }
    )

    assert bm25_factory.retriever.queries == [
        "未出库订单修改地址条件",
        "待审核售后单材料要求",
        "退款规则",
        "售后退款",
    ]
    assert [
        item["doc_type"] for item in bm25_factory.calls
    ] == [None, None, "RULE"]
    assert {
        tuple(item["metadata"]["sub_query_ids"])
        for item in result["reranked_documents"]
    } == {("SQ1",), ("SQ2",)}
    assert result["sub_query_coverage"]["coverage_rate"] == 1.0
    assert rerank_service.calls[0]["top_n"] == 4
    assert result["retrieval_round"] == 1
    assert context_service.calls[0]["sub_queries"] == [
        {
            "sub_query_id": "SQ1",
            "question": "未出库订单修改地址条件",
            "target_doc_types": ["FAQ"],
        },
        {
            "sub_query_id": "SQ2",
            "question": "待审核售后单材料要求",
            "target_doc_types": ["SOP"],
        },
    ]


async def test_dependent_query_executes_second_hop_with_intermediate_fact() -> None:
    bm25_factory = SequencedRetrieverFactory()
    context_service = FakeContextBuildService()
    rerank_service = FakeRerankService()
    extractor = FakeIntermediateFactExtractor(
        IntermediateFactResult(
            success=True,
            intermediate_fact="高风险订单",
            evidence_quote="订单命中高风险规则",
            supporting_chunk_id=1,
            confidence=0.93,
            reason="第一跳识别出风险等级",
        )
    )
    decomposer = FakeQueryDecomposer(
        QueryDecompositionResult(
            requires_decomposition=True,
            decomposition_type="DEPENDENT",
            benefit_score=0.95,
            reason="审批人查询依赖风险等级",
            sub_queries=[
                DecomposedSubQuery(
                    sub_query_id="SQ1",
                    question="订单对应什么风险等级？",
                    target_doc_types=["RULE"],
                ),
                DecomposedSubQuery(
                    sub_query_id="SQ2",
                    question="{{intermediate_fact}}由谁审批？",
                    target_doc_types=["SOP"],
                    depends_on_sub_query_id="SQ1",
                    is_template=True,
                ),
            ],
        )
    )
    quality_policy = SequencedQualityPolicy(
        [
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.GOOD,
                score=0.9,
                retry_strategy=RetryStrategy.NONE,
                reasons=["第一跳证据可用"],
            ),
            RetrievalQualityDecision(
                level=RetrievalQualityLevel.GOOD,
                score=0.9,
                retry_strategy=RetryStrategy.NONE,
                reasons=["两跳证据充分"],
            ),
        ]
    )
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
        rerank_enabled=True,
        rerank_top_n=3,
        query_decomposition_enabled=True,
        query_decomposition_rerank_extra_limit=2,
        sub_query_min_candidates=1,
        sub_query_rerank_quota=1,
        dependent_multi_hop_enabled=True,
        dependent_multi_hop_max_hops=2,
    )
    graph = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        query_decomposer=decomposer,
        intermediate_fact_extractor=extractor,
        rerank_service=rerank_service,
        context_build_service=context_service,
        bm25_retriever_factory=bm25_factory,
        quality_policy=quality_policy,
    ).build()

    result = await graph.ainvoke(
        {
            "question": "这个订单是什么风险等级，对应由谁审批？",
            "session_id": "session-1",
            "business_domain": "after_sales",
            "trace_id": "trace-1",
            "top_k": 5,
            "status": "STARTED",
        }
    )

    assert result["retrieval_round"] == 2
    assert len(result["retrieval_attempts"]) == 2
    assert result["retrieval_attempts"][1]["strategy"] == (
        "DEPENDENT_HOP"
    )
    assert "高风险订单由谁审批？" in (
        bm25_factory.retriever.queries
    )
    assert result["dependent_hop"]["status"] == "COMPLETED"
    assert result["dependent_hop"]["intermediate_fact"] == (
        "高风险订单"
    )
    assert result["dependent_hop"]["supporting_chunk_id"] == 1
    assert result["decomposition"]["sub_queries"][1][
        "question"
    ] == "高风险订单由谁审批？"
    assert result["sub_query_coverage"]["coverage_rate"] == 1.0
    assert context_service.calls[0]["sub_queries"][1][
        "question"
    ] == "高风险订单由谁审批？"


async def test_dependent_fact_failure_falls_back_to_original_question() -> None:
    extractor = FakeIntermediateFactExtractor(
        IntermediateFactResult(
            success=False,
            reason="第一跳证据无法确定风险等级",
        )
    )
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
        dependent_multi_hop_enabled=True,
    )
    builder = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        intermediate_fact_extractor=extractor,
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
    )

    result = await builder.prepare_dependent_hop(
        {
            "question": "这个订单是什么风险等级，对应由谁审批？",
            "retrieval_round": 1,
            "decomposition": {
                "requires_decomposition": True,
                "decomposition_type": "DEPENDENT",
                "sub_queries": [
                    {
                        "sub_query_id": "SQ1",
                        "question": "订单对应什么风险等级？",
                    },
                    {
                        "sub_query_id": "SQ2",
                        "question": (
                            "{{intermediate_fact}}由谁审批？"
                        ),
                        "depends_on_sub_query_id": "SQ1",
                        "is_template": True,
                    },
                ],
            },
            "dependent_hop": {
                "enabled": True,
                "status": "FIRST_HOP_READY",
                "current_hop": 1,
            },
            "reranked_documents": [
                {
                    "chunk_id": 1,
                    "page_content": "没有明确风险等级。",
                }
            ],
        }
    )

    assert result["retrieval_round"] == 2
    assert result["retrieval_queries"] == [
        "这个订单是什么风险等级，对应由谁审批？"
    ]
    assert result["dependent_hop"]["fallback_used"] is True
    assert result["dependent_hop"]["status"] == (
        "SECOND_HOP_FALLBACK"
    )


async def test_decomposer_exception_falls_back_to_original_retrieval() -> None:
    builder = RagRetrievalGraphBuilder(
        query_understanding_service=FakeQueryUnderstandingService(),
        query_decomposer=FakeQueryDecomposer(
            error=RuntimeError("decomposer unavailable")
        ),
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
    )
    state = {
        "question": "订单怎么改地址，同时售后单要补什么材料？",
        "rewritten_question": "退款规则",
        "retrieval_queries": ["退款规则", "售后退款"],
        "target_doc_types": ["RULE"],
    }

    result = await builder.decompose_query(state)

    assert result["decomposition"]["requires_decomposition"] is False
    assert result["decomposition"]["fallback_used"] is True
    assert result["retrieval_queries"] == ["退款规则", "售后退款"]
    assert result["retrieval_tasks"] == []


async def test_prepare_retry_rewrites_only_uncovered_sub_queries() -> None:
    query_rewriter = FakeQueryRewriter()
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
    )
    builder = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
        query_rewriter=query_rewriter,
    )

    result = await builder.prepare_retry(
        {
            "question": "订单怎么改地址，同时售后单要补什么材料？",
            "retrieval_round": 1,
            "retrieval_queries": ["地址条件", "材料要求"],
            "retrieval_tasks": [
                {
                    "sub_query_id": "SQ1",
                    "question": "地址条件",
                    "target_doc_types": ["FAQ"],
                },
                {
                    "sub_query_id": "SQ2",
                    "question": "材料要求",
                    "target_doc_types": ["SOP"],
                },
            ],
            "decomposition": {
                "requires_decomposition": True,
                "sub_queries": [],
            },
            "sub_query_coverage": {
                "coverage_rate": 0.5,
                "items": {
                    "SQ1": {"covered": True},
                    "SQ2": {"covered": False},
                },
            },
            "target_doc_types": [],
            "business_domain": "after_sales",
            "merged_documents": [],
            "retrieval_quality": {
                "retry_strategy": "QUERY_REWRITE",
                "reasons": ["子问题证据覆盖不足"],
            },
        }
    )

    assert len(query_rewriter.calls) == 1
    assert query_rewriter.calls[0]["original_question"] == "材料要求"
    assert result["retrieval_tasks"] == [
        {
            "sub_query_id": "SQ2",
            "question": "订单取消规则 新版 旧版",
            "target_doc_types": ["SOP"],
        }
    ]
    assert result["retrieval_queries"] == ["订单取消规则 新版 旧版"]
    assert result["query_variant"] == "SUB_QUERY_REWRITTEN"


async def test_prepare_retry_forces_bm25_or_relaxes_doc_type() -> None:
    settings = SimpleNamespace(
        adaptive_retrieval_enabled=True,
        adaptive_max_rounds=2,
        rrf_rank_constant=60,
    )
    builder = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=FakeQueryUnderstandingService(),
        rerank_service=FakeRerankService(),
        context_build_service=FakeContextBuildService(),
        query_rewriter=FakeQueryRewriter(),
    )
    base_state = {
        "question": "错误码 F-ORDER-001 怎么处理？",
        "retrieval_round": 1,
        "retrieval_queries": ["订单错误处理"],
        "target_doc_types": ["RULE"],
        "business_domain": "after_sales",
        "merged_documents": [],
        "retrieval_quality": {
            "retry_strategy": "FORCE_BM25",
            "reasons": ["精确词缺失"],
        },
    }

    forced = await builder.prepare_retry(base_state)

    assert forced["retrieval_mode"] == "bm25"
    assert forced["retrieval_round"] == 2
    assert "F-ORDER-001" in forced["retrieval_queries"][0]
    assert forced["target_doc_types"] == []
    assert forced["removed_filters"] == ["doc_type"]

    relaxed = await builder.prepare_retry(
        {
            **base_state,
            "retrieval_quality": {
                "retry_strategy": "RELAX_FILTER",
                "reasons": ["目标类型未覆盖"],
            },
        }
    )

    assert relaxed["target_doc_types"] == []
    assert relaxed["business_domain"] == "after_sales"
    assert relaxed["removed_filters"] == ["doc_type"]
