from types import SimpleNamespace

from src.rag_platform.application.context_build_service import ContextBuildService
from src.rag_platform.application.query_understanding_service import (
    QueryUnderstandingService,
)
from src.rag_platform.application.rerank_service import RerankService
from src.rag_platform.domain.context import ContextBuildResult, ContextBuildStatus
from src.rag_platform.domain.rerank import RerankResultItem
from src.rag_platform.schemas.query_analysis import (
    QueryAnalysisRequest,
    QueryAnalysisResult,
)


class FakeQueryAnalyzer:
    def analyze(self, query, business_domain):
        return QueryAnalysisResult(
            original_query=query,
            rewritten_query="退款规则",
            target_doc_types=["RULE"],
            retrieval_mode="bm25",
            business_domain=business_domain,
            confidence=0.8,
            reason="规则测试",
        )


class FakeQueryRepository:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    def save_analysis_log(self, **kwargs) -> None:
        self.saved.append(kwargs)


async def test_query_understanding_service_uses_injected_dependencies() -> None:
    repository = FakeQueryRepository()
    service = QueryUnderstandingService(
        settings=SimpleNamespace(query_analysis_use_llm=False),
        rule_analyzer=FakeQueryAnalyzer(),
        repository=repository,
    )

    response = await service.analyze(
        QueryAnalysisRequest(
            query="怎么退款？",
            session_id="session-1",
            business_domain="after_sales",
        )
    )

    assert response.result.rewritten_query == "退款规则"
    assert repository.saved[0]["session_id"] == "session-1"
    assert repository.saved[0]["result"] is response.result


class FakeReranker:
    calls: list[dict] = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
        )
        return [
            RerankResultItem(
                chunk_id=1,
                document_index=0,
                relevance_score=0.95,
                after_rank=1,
                text="退款规则",
                metadata={"page_content": "退款规则"},
            )
        ]


class FakeRerankRepository:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.saved: list[dict] = []

    def create_rerank_log(self, **kwargs) -> int:
        self.created.append(kwargs)
        return 7

    def save_rerank_items(self, *args) -> None:
        self.saved.append({"args": args})


async def test_rerank_service_uses_injected_reranker_and_repository() -> None:
    FakeReranker.calls = []
    repository = FakeRerankRepository()
    service = RerankService(
        settings=SimpleNamespace(
            rerank_enabled=True,
            rerank_provider="fake",
            rerank_model="fake-reranker",
            rerank_top_n=5,
            rerank_min_score=0.0,
            rerank_fail_open=False,
        ),
        repository=repository,
        reranker_factory=FakeReranker,
    )

    documents, info = await service.rerank_documents(
        trace_id="trace-1",
        query="退款",
        documents=[{"chunk_id": 1, "page_content": "退款规则"}],
        top_n=7,
    )

    assert documents[0]["chunk_id"] == 1
    assert info["status"] == "SUCCESS"
    assert repository.created[0]["provider"] == "fake"
    assert repository.created[0]["top_n"] == 7
    assert repository.saved[0]["args"][0] == 7
    assert FakeReranker.calls[0]["top_n"] == 7


class FakeContextBuilder:
    def build(self, documents):
        return ContextBuildResult(
            context="退款规则正文",
            citations=[],
            used_chunks=[],
            estimated_tokens=12,
            status=ContextBuildStatus.SUCCESS,
            message="ok",
        )


class FakeContextRepository:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.saved: list[dict] = []

    def create_context_log(self, **kwargs) -> int:
        self.created.append(kwargs)
        return 8

    def save_citation_logs(self, **kwargs) -> None:
        self.saved.append(kwargs)


def test_context_build_service_uses_injected_builder_and_repository() -> None:
    repository = FakeContextRepository()
    service = ContextBuildService(
        settings=SimpleNamespace(
            context_max_tokens=1000,
            context_expand_parent=False,
            context_expand_previous_next=False,
            context_expand_same_section=False,
        ),
        builder=FakeContextBuilder(),
        repository=repository,
    )

    result, info = service.build_context(
        trace_id="trace-1",
        query_text="退款",
        documents=[{"chunk_id": 1}],
    )

    assert result.context == "退款规则正文"
    assert info["context_log_id"] == 8
    assert repository.created[0]["estimated_tokens"] == 12
    assert repository.saved[0]["context_log_id"] == 8
