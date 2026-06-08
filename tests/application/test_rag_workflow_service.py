from src.rag_platform.application.rag_workflow_service import RagWorkflowService
from src.rag_platform.schemas.rag_workflow import RagRetrievalWorkflowRequest


class FakeGraph:
    def __init__(self) -> None:
        self.received_state: dict | None = None

    async def ainvoke(self, state: dict) -> dict:
        self.received_state = state
        return {
            **state,
            "rewritten_question": "退款规则",
            "retrieval_mode": "hybrid",
            "target_doc_types": ["RULE"],
            "status": "CONTEXT_READY",
            "retrieval_quality": {"quality": "GOOD"},
            "context": "退款规则正文",
            "citations": [{"citation_id": "C1", "chunk_id": 101}],
        }


async def test_run_retrieval_workflow_uses_injected_graph() -> None:
    graph = FakeGraph()
    service = RagWorkflowService(graph=graph)

    response = await service.run_retrieval_workflow(
        RagRetrievalWorkflowRequest(
            question="怎么退款？",
            session_id="session-1",
            business_domain="after_sales",
            top_k=5,
        )
    )

    assert graph.received_state is not None
    assert graph.received_state["question"] == "怎么退款？"
    assert graph.received_state["session_id"] == "session-1"
    assert graph.received_state["business_domain"] == "after_sales"
    assert graph.received_state["top_k"] == 5
    assert graph.received_state["trace_id"]
    assert response.trace_id == graph.received_state["trace_id"]
    assert response.rewritten_question == "退款规则"
    assert response.status == "CONTEXT_READY"
    assert response.context == "退款规则正文"
