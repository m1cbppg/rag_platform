from src.rag_platform.application.rag_workflow_service import RagWorkflowService
from src.rag_platform.schemas.rag_workflow import RagRetrievalWorkflowRequest


class FakeGraph:
    def __init__(self) -> None:
        self.received_state: dict | None = None

    async def ainvoke(self, state: dict) -> dict:
        self.received_state = state
        return {
            **state,
            "query_analysis": {
                "confidence": 0.9,
                "need_clarification": True,
                "clarification_question": "订单是否已经发货？",
            },
            "rewritten_question": "退款规则",
            "retrieval_mode": "hybrid",
            "target_doc_types": ["RULE"],
            "decomposition": {
                "requires_decomposition": True,
                "reason": "包含两个信息需求",
                "sub_queries": [
                    {
                        "sub_query_id": "SQ1",
                        "question": "退款条件？",
                    },
                    {
                        "sub_query_id": "SQ2",
                        "question": "退款时限？",
                    },
                ],
            },
            "sub_query_coverage": {
                "total_sub_queries": 2,
                "covered_sub_queries": 2,
                "coverage_rate": 1.0,
                "items": {},
            },
            "need_clarification": True,
            "clarification_question": "订单是否已经发货？",
            "status": "CONTEXT_READY",
            "retrieval_quality": {"quality": "GOOD"},
            "retrieval_round": 2,
            "max_retrieval_rounds": 2,
            "retrieval_attempts": [
                {
                    "round_no": 1,
                    "strategy": "INITIAL",
                    "queries": ["退款规则"],
                },
                {
                    "round_no": 2,
                    "strategy": "QUERY_REWRITE",
                    "queries": ["售后退款规则"],
                },
            ],
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
    assert response.query_analysis["confidence"] == 0.9
    assert response.need_clarification is True
    assert response.clarification_question == "订单是否已经发货？"
    assert response.status == "CONTEXT_READY"
    assert response.context == "退款规则正文"
    assert response.retrieval_round == 2
    assert response.max_retrieval_rounds == 2
    assert len(response.retrieval_attempts) == 2
    assert response.retrieval_attempts[1]["strategy"] == "QUERY_REWRITE"
    assert response.decomposition["requires_decomposition"] is True
    assert response.sub_query_coverage["coverage_rate"] == 1.0
