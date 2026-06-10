from dataclasses import dataclass
from typing import Any, Protocol

from src.rag_platform.domain.answer import AnswerStatus
from src.rag_platform.evaluation.models import ActualAction
from src.rag_platform.schemas.chat_execution import ChatExecutionResult
from src.rag_platform.schemas.chat_v2 import ChatRequestV2


class ObservableChatService(Protocol):
    async def execute(
        self,
        request: ChatRequestV2,
    ) -> ChatExecutionResult: ...


@dataclass(frozen=True)
class RagEvaluationObservation:
    trace_id: str
    answer: str
    actual_action: ActualAction
    retrieved_chunk_ids: list[int]
    cited_chunk_ids: list[int]
    citations: list[dict[str, Any]]
    context: str
    context_blocks: list[str]
    retrieval_hits: list[dict[str, Any]]
    latency_ms: int


class ChatServiceEvaluationAdapter:
    def __init__(self, chat_service: ObservableChatService) -> None:
        self.chat_service = chat_service

    async def run(
        self,
        *,
        question: str,
        session_id: str | None = None,
        business_domain: str | None = None,
        top_k: int = 20,
    ) -> RagEvaluationObservation:
        execution = await self.chat_service.execute(
            ChatRequestV2(
                question=question,
                session_id=session_id,
                business_domain=business_domain,
                top_k=top_k,
            )
        )
        response = execution.response
        workflow = execution.workflow
        final_chunk_ids = [
            int(citation["chunk_id"])
            for citation in workflow.citations
            if citation.get("chunk_id") is not None
        ]
        used_citation_ids = set(
            response.citation_validation.get(
                "used_citation_ids",
                [],
            )
        )
        cited_chunk_ids = [
            int(citation["chunk_id"])
            for citation in response.citations
            if citation.get("citation_id") in used_citation_ids
            and citation.get("chunk_id") is not None
        ]
        return RagEvaluationObservation(
            trace_id=response.trace_id,
            answer=response.answer,
            actual_action=_actual_action(response.status),
            retrieved_chunk_ids=final_chunk_ids,
            cited_chunk_ids=cited_chunk_ids,
            citations=response.citations,
            context=workflow.context or "",
            context_blocks=_context_blocks(workflow.context or ""),
            retrieval_hits=_retrieval_hits(
                question=question,
                execution=execution,
            ),
            latency_ms=execution.latency_ms,
        )


def _context_blocks(context: str) -> list[str]:
    return [
        block.strip()
        for block in context.split("\n\n---\n\n")
        if block.strip()
    ]


def _actual_action(status: str) -> ActualAction:
    if status == AnswerStatus.SUCCESS.value:
        return ActualAction.ANSWER
    if status == AnswerStatus.REFUSED.value:
        return ActualAction.REFUSE
    return ActualAction.ERROR


def _retrieval_hits(
    *,
    question: str,
    execution: ChatExecutionResult,
) -> list[dict[str, Any]]:
    workflow = execution.workflow
    query_text = workflow.rewritten_question or question
    hits: list[dict[str, Any]] = []
    merged_channel = (workflow.retrieval_mode or "HYBRID").upper()
    for rank, document in enumerate(workflow.documents, start=1):
        is_hybrid = merged_channel == "HYBRID"
        hits.append(
            {
                "retrieval_round": 1,
                "query_variant": "ORIGINAL",
                "query_text": query_text,
                "channel": merged_channel,
                "chunk_id": document.chunk_id,
                "rank_no": rank,
                "raw_score": None if is_hybrid else document.score,
                "fused_score": document.score if is_hybrid else None,
                "metadata": {
                    **document.metadata,
                    "source": document.source,
                },
            }
        )
    for rank, document in enumerate(
        workflow.reranked_documents,
        start=1,
    ):
        hits.append(
            {
                "retrieval_round": 1,
                "query_variant": "ORIGINAL",
                "query_text": query_text,
                "channel": "RERANK",
                "chunk_id": document.chunk_id,
                "rank_no": rank,
                "raw_score": document.score,
                "rerank_score": document.rerank_score,
                "metadata": document.metadata,
            }
        )
    for rank, citation in enumerate(workflow.citations, start=1):
        if citation.get("chunk_id") is None:
            continue
        hits.append(
            {
                "retrieval_round": 1,
                "query_variant": "ORIGINAL",
                "query_text": query_text,
                "channel": "FINAL",
                "chunk_id": int(citation["chunk_id"]),
                "rank_no": rank,
                "metadata": {
                    "citation_id": citation.get("citation_id"),
                    "expansion_type": citation.get("expansion_type"),
                },
            }
        )
    return hits
