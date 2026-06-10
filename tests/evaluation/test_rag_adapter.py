from types import SimpleNamespace

from src.rag_platform.evaluation.models import ActualAction
from src.rag_platform.evaluation.rag_adapter import (
    ChatServiceEvaluationAdapter,
)
from src.rag_platform.schemas.chat_execution import ChatExecutionResult
from src.rag_platform.schemas.chat_v2 import ChatResponseV2
from src.rag_platform.schemas.rag_workflow import (
    RagRetrievalWorkflowResponse,
    WorkflowDocumentResponse,
)


class FakeChatService:
    async def execute(self, request):
        workflow = RagRetrievalWorkflowResponse(
            trace_id="trace-1",
            question=request.question,
            rewritten_question="退款规则",
            retrieval_mode="hybrid",
            status="CONTEXT_READY",
            documents=[
                WorkflowDocumentResponse(
                    chunk_id=101,
                    score=0.8,
                    source="hybrid",
                    content="规则A",
                    metadata={
                        "sources": ["vector", "bm25"],
                        "vector_rank": 2,
                        "vector_raw_score": 0.82,
                        "bm25_rank": 1,
                        "bm25_raw_score": 7.5,
                    },
                )
            ],
            reranked_documents=[
                WorkflowDocumentResponse(
                    chunk_id=102,
                    score=0.7,
                    rerank_score=0.95,
                    after_rank=1,
                    content="规则B",
                ),
                WorkflowDocumentResponse(
                    chunk_id=101,
                    score=0.8,
                    rerank_score=0.9,
                    after_rank=2,
                    content="规则A",
                ),
            ],
            context="[C1]规则B\n[C2]规则A",
            citations=[
                {"citation_id": "C1", "chunk_id": 102},
                {"citation_id": "C2", "chunk_id": 101},
            ],
        )
        response = ChatResponseV2(
            trace_id="trace-1",
            question=request.question,
            rewritten_question="退款规则",
            answer="应按规则B处理。[C1]",
            status="SUCCESS",
            citations=workflow.citations,
            citation_validation={
                "used_citation_ids": ["C1"],
            },
        )
        return ChatExecutionResult(
            response=response,
            workflow=workflow,
            latency_ms=123,
        )


async def test_adapter_extracts_final_chunks_and_actual_citations() -> None:
    observation = await ChatServiceEvaluationAdapter(
        FakeChatService()
    ).run(
        question="怎么退款？",
        business_domain="ecommerce_after_sales",
        top_k=5,
    )

    assert observation.actual_action == ActualAction.ANSWER
    assert observation.retrieved_chunk_ids == [102, 101]
    assert observation.cited_chunk_ids == [102]
    assert observation.context == "[C1]规则B\n[C2]规则A"
    assert observation.context_blocks == ["[C1]规则B\n[C2]规则A"]
    assert observation.latency_ms == 123
    assert [hit["channel"] for hit in observation.retrieval_hits] == [
        "HYBRID",
        "RERANK",
        "RERANK",
        "FINAL",
        "FINAL",
    ]
    merged_hit = observation.retrieval_hits[0]
    assert merged_hit["raw_score"] is None
    assert merged_hit["fused_score"] == 0.8
    assert merged_hit["metadata"]["sources"] == ["vector", "bm25"]
    assert merged_hit["metadata"]["bm25_rank"] == 1
    assert merged_hit["metadata"]["vector_rank"] == 2


async def test_adapter_maps_refused_status_without_inventing_clarify() -> None:
    service = FakeChatService()
    original_execute = service.execute

    async def refused(request):
        execution = await original_execute(request)
        return execution.model_copy(
            update={
                "response": execution.response.model_copy(
                    update={
                        "status": "REFUSED",
                        "answer": "知识库信息不足。",
                        "citation_validation": {},
                    }
                )
            }
        )

    service.execute = refused
    observation = await ChatServiceEvaluationAdapter(service).run(
        question="未知规则？"
    )

    assert observation.actual_action == ActualAction.REFUSE
    assert observation.cited_chunk_ids == []
