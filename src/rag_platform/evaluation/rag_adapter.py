from dataclasses import dataclass, field
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
    retrieval_rounds: int
    latency_ms: int
    decomposition: dict[str, Any] = field(default_factory=dict)
    sub_query_coverage: dict[str, Any] = field(
        default_factory=dict
    )
    dependent_hop: dict[str, Any] = field(default_factory=dict)


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
            retrieval_rounds=workflow.retrieval_round,
            latency_ms=execution.latency_ms,
            decomposition=workflow.decomposition,
            sub_query_coverage=workflow.sub_query_coverage,
            dependent_hop=workflow.dependent_hop,
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
    if status == AnswerStatus.CLARIFIED.value:
        return ActualAction.CLARIFY
    return ActualAction.ERROR


def _retrieval_hits(
    *,
    question: str,
    execution: ChatExecutionResult,
) -> list[dict[str, Any]]:
    workflow = execution.workflow
    query_text = workflow.rewritten_question or question
    hits: list[dict[str, Any]] = []
    if workflow.retrieval_attempts:
        for attempt in workflow.retrieval_attempts:
            hits.extend(
                _attempt_hits(
                    attempt=attempt,
                    fallback_query=query_text,
                )
            )
        hits.extend(
            _final_hits(
                workflow=workflow,
                retrieval_round=workflow.retrieval_round,
                query_text=_last_attempt_query(
                    workflow.retrieval_attempts,
                    query_text,
                ),
            )
        )
        return hits

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


def _attempt_hits(
    *,
    attempt: dict[str, Any],
    fallback_query: str,
) -> list[dict[str, Any]]:
    round_no = int(attempt.get("round_no") or 1)
    query_variant = str(
        attempt.get("query_variant") or "ORIGINAL"
    )
    queries = list(attempt.get("queries") or [])
    query_text = str(
        queries[0] if queries else fallback_query
    )
    result: list[dict[str, Any]] = []
    default_channel = str(
        attempt.get("retrieval_mode") or "hybrid"
    ).upper()
    for rank, document in enumerate(
        attempt.get("documents") or [],
        start=1,
    ):
        channel = str(
            document.get("source") or default_channel
        ).upper()
        score = document.get("score")
        is_fused = channel in {"HYBRID", "ADAPTIVE"}
        result.append(
            {
                "retrieval_round": round_no,
                "query_variant": query_variant,
                "query_text": query_text,
                "channel": channel,
                "chunk_id": int(document["chunk_id"]),
                "rank_no": rank,
                "raw_score": None if is_fused else score,
                "fused_score": score if is_fused else None,
                "metadata": {
                    **(document.get("metadata") or {}),
                    "strategy": attempt.get("strategy"),
                    "queries": queries,
                    "removed_filters": attempt.get(
                        "removed_filters",
                        [],
                    ),
                },
            }
        )
    for rank, document in enumerate(
        attempt.get("reranked_documents") or [],
        start=1,
    ):
        result.append(
            {
                "retrieval_round": round_no,
                "query_variant": query_variant,
                "query_text": query_text,
                "channel": "RERANK",
                "chunk_id": int(document["chunk_id"]),
                "rank_no": rank,
                "raw_score": document.get("score"),
                "rerank_score": (
                    document.get("rerank_score")
                    or (document.get("metadata") or {}).get(
                        "rerank_score"
                    )
                ),
                "metadata": {
                    **(document.get("metadata") or {}),
                    "strategy": attempt.get("strategy"),
                },
            }
        )
    return result


def _final_hits(
    *,
    workflow,
    retrieval_round: int,
    query_text: str,
) -> list[dict[str, Any]]:
    result = []
    metadata_by_chunk_id = {
        int(document.chunk_id): document.metadata
        for document in workflow.reranked_documents
    }
    for rank, citation in enumerate(workflow.citations, start=1):
        if citation.get("chunk_id") is None:
            continue
        result.append(
            {
                "retrieval_round": retrieval_round,
                "query_variant": "FINAL",
                "query_text": query_text,
                "channel": "FINAL",
                "chunk_id": int(citation["chunk_id"]),
                "rank_no": rank,
                "metadata": {
                    **metadata_by_chunk_id.get(
                        int(citation["chunk_id"]),
                        {},
                    ),
                    "citation_id": citation.get("citation_id"),
                    "expansion_type": citation.get(
                        "expansion_type"
                    ),
                },
            }
        )
    return result


def _last_attempt_query(
    attempts: list[dict[str, Any]],
    fallback_query: str,
) -> str:
    queries = list(attempts[-1].get("queries") or [])
    return str(queries[0] if queries else fallback_query)
