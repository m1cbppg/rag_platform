from langchain_core.documents import Document

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


class FakeRetrieverFactory:
    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.retriever = FakeRetriever()

    def __call__(self, **kwargs) -> FakeRetriever:
        self.kwargs = kwargs
        return self.retriever


class FakeRerankService:
    async def rerank_documents(self, trace_id, query, documents):
        return documents, {"status": "SUCCESS", "trace_id": trace_id, "query": query}


class FakeContextBuildService:
    def build_context(self, trace_id, query_text, documents):
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
