import json
import time
from collections.abc import AsyncGenerator

from src.rag_platform.application.rag_workflow_service import RagWorkflowService
from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.answer import AnswerStatus, ChatStreamEventType
from src.rag_platform.infrastructure.repositories.answer_repository import AnswerRepository
from src.rag_platform.rag.answer.citation_validator import CitationValidator
from src.rag_platform.rag.answer.deepseek_answer_generator import DeepSeekAnswerGenerator
from src.rag_platform.schemas.chat_v2 import ChatRequestV2, ChatResponseV2
from src.rag_platform.schemas.rag_workflow import RagRetrievalWorkflowRequest


class ChatService:
    """
    正式 RAG Chat 服务。

    职责：
    1. 执行 RAG 工作流，拿到 context + citations；
    2. 检查上下文是否可用；
    3. 调用 DeepSeek 生成答案；
    4. 校验引用；
    5. 保存 answer_log；
    6. 支持 SSE 流式输出。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.workflow_service = RagWorkflowService()
        self.answer_generator = DeepSeekAnswerGenerator()
        self.answer_repository = AnswerRepository()
        self.citation_validator = CitationValidator()

    async def chat(
        self,
        request: ChatRequestV2,
    ) -> ChatResponseV2:
        start_time = time.perf_counter()

        workflow_response = await self.workflow_service.run_retrieval_workflow(
            RagRetrievalWorkflowRequest(
                question=request.question,
                session_id=request.session_id,
                business_domain=request.business_domain,
                top_k=request.top_k,
            )
        )

        trace_id = workflow_response.trace_id

        context = workflow_response.context or ""
        citations = workflow_response.citations or []

        if self._should_refuse_due_to_empty_context(context):
            answer = "当前知识库中没有足够信息回答该问题，建议补充相关 FAQ、SOP、业务规则或操作手册后再查询。"

            answer_log_id = self._save_refused_answer(
                request=request,
                workflow_response=workflow_response,
                answer=answer,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
            )

            return ChatResponseV2(
                trace_id=trace_id,
                answer_log_id=answer_log_id,
                question=request.question,
                rewritten_question=workflow_response.rewritten_question,
                answer=answer,
                status=AnswerStatus.REFUSED.value,
                citations=citations,
                retrieval_quality=workflow_response.retrieval_quality,
                rerank_info=workflow_response.rerank_info,
                context_build_info=workflow_response.context_build_info,
            )

        answer_log_id = self.answer_repository.create_answer_log(
            trace_id=trace_id,
            session_id=request.session_id,
            question=request.question,
            rewritten_question=workflow_response.rewritten_question,
            model=self.settings.answer_model,
            temperature=self.settings.answer_temperature,
            max_tokens=self.settings.answer_max_tokens,
            context_log_id=workflow_response.context_build_info.get("context_log_id"),
            context_tokens=workflow_response.context_build_info.get("estimated_tokens"),
            citation_count=len(citations),
            status=AnswerStatus.STREAMING.value,
        )

        self.answer_repository.save_answer_citations(
            answer_log_id=answer_log_id,
            trace_id=trace_id,
            citations=citations,
        )

        try:
            answer = await self.answer_generator.generate(
                question=request.question,
                rewritten_question=workflow_response.rewritten_question,
                context=context,
                citations=citations,
            )

            citation_validation = self.citation_validator.validate(
                answer=answer,
                citations=citations,
                require_citation=self.settings.answer_require_citation,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            self.answer_repository.update_answer_log(
                answer_log_id=answer_log_id,
                answer=answer,
                status=AnswerStatus.SUCCESS.value,
                latency_ms=latency_ms,
                error_message=None,
            )

            return ChatResponseV2(
                trace_id=trace_id,
                answer_log_id=answer_log_id,
                question=request.question,
                rewritten_question=workflow_response.rewritten_question,
                answer=answer,
                status=AnswerStatus.SUCCESS.value,
                citations=citations,
                citation_validation=citation_validation,
                retrieval_quality=workflow_response.retrieval_quality,
                rerank_info=workflow_response.rerank_info,
                context_build_info=workflow_response.context_build_info,
            )

        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            self.answer_repository.update_answer_log(
                answer_log_id=answer_log_id,
                answer=None,
                status=AnswerStatus.FAILED.value,
                latency_ms=latency_ms,
                error_message=str(exc),
            )

            raise

    async def stream_chat(
        self,
        request: ChatRequestV2,
    ) -> AsyncGenerator[str, None]:
        """
        SSE 流式 Chat。

        SSE 每条消息格式：
            event: xxx
            data: {...}

        最后空行结束一条事件。
        """

        start_time = time.perf_counter()
        full_answer_parts: list[str] = []

        workflow_response = await self.workflow_service.run_retrieval_workflow(
            RagRetrievalWorkflowRequest(
                question=request.question,
                session_id=request.session_id,
                business_domain=request.business_domain,
                top_k=request.top_k,
            )
        )

        trace_id = workflow_response.trace_id
        context = workflow_response.context or ""
        citations = workflow_response.citations or []

        yield self._sse(
            event=ChatStreamEventType.TRACE.value,
            data={
                "trace_id": trace_id,
            },
        )

        yield self._sse(
            event=ChatStreamEventType.RETRIEVAL.value,
            data={
                "status": workflow_response.status,
                "retrieval_quality": workflow_response.retrieval_quality,
                "rerank_info": workflow_response.rerank_info,
            },
        )

        yield self._sse(
            event=ChatStreamEventType.CONTEXT.value,
            data={
                "context_build_info": workflow_response.context_build_info,
                "citations": citations,
            },
        )

        if self._should_refuse_due_to_empty_context(context):
            answer = "当前知识库中没有足够信息回答该问题，建议补充相关 FAQ、SOP、业务规则或操作手册后再查询。"

            answer_log_id = self._save_refused_answer(
                request=request,
                workflow_response=workflow_response,
                answer=answer,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
            )

            yield self._sse(
                event=ChatStreamEventType.DELTA.value,
                data={"text": answer},
            )

            yield self._sse(
                event=ChatStreamEventType.DONE.value,
                data={
                    "trace_id": trace_id,
                    "answer_log_id": answer_log_id,
                    "status": AnswerStatus.REFUSED.value,
                },
            )
            return

        answer_log_id = self.answer_repository.create_answer_log(
            trace_id=trace_id,
            session_id=request.session_id,
            question=request.question,
            rewritten_question=workflow_response.rewritten_question,
            model=self.settings.answer_model,
            temperature=self.settings.answer_temperature,
            max_tokens=self.settings.answer_max_tokens,
            context_log_id=workflow_response.context_build_info.get("context_log_id"),
            context_tokens=workflow_response.context_build_info.get("estimated_tokens"),
            citation_count=len(citations),
            status=AnswerStatus.STREAMING.value,
        )

        self.answer_repository.save_answer_citations(
            answer_log_id=answer_log_id,
            trace_id=trace_id,
            citations=citations,
        )

        try:
            async for delta in self.answer_generator.stream_generate(
                question=request.question,
                rewritten_question=workflow_response.rewritten_question,
                context=context,
                citations=citations,
            ):
                full_answer_parts.append(delta)

                yield self._sse(
                    event=ChatStreamEventType.DELTA.value,
                    data={"text": delta},
                )

            full_answer = "".join(full_answer_parts)

            citation_validation = self.citation_validator.validate(
                answer=full_answer,
                citations=citations,
                require_citation=self.settings.answer_require_citation,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            self.answer_repository.update_answer_log(
                answer_log_id=answer_log_id,
                answer=full_answer,
                status=AnswerStatus.SUCCESS.value,
                latency_ms=latency_ms,
                error_message=None,
            )

            yield self._sse(
                event=ChatStreamEventType.DONE.value,
                data={
                    "trace_id": trace_id,
                    "answer_log_id": answer_log_id,
                    "status": AnswerStatus.SUCCESS.value,
                    "citation_validation": citation_validation,
                },
            )

        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            self.answer_repository.update_answer_log(
                answer_log_id=answer_log_id,
                answer="".join(full_answer_parts) or None,
                status=AnswerStatus.FAILED.value,
                latency_ms=latency_ms,
                error_message=str(exc),
            )

            yield self._sse(
                event=ChatStreamEventType.ERROR.value,
                data={
                    "trace_id": trace_id,
                    "answer_log_id": answer_log_id,
                    "error": str(exc),
                },
            )

    def _should_refuse_due_to_empty_context(self, context: str) -> bool:
        return self.settings.answer_fail_when_context_empty and not context.strip()

    def _save_refused_answer(
        self,
        request: ChatRequestV2,
        workflow_response,
        answer: str,
        latency_ms: int,
    ) -> int:
        answer_log_id = self.answer_repository.create_answer_log(
            trace_id=workflow_response.trace_id,
            session_id=request.session_id,
            question=request.question,
            rewritten_question=workflow_response.rewritten_question,
            model=self.settings.answer_model,
            temperature=self.settings.answer_temperature,
            max_tokens=self.settings.answer_max_tokens,
            context_log_id=workflow_response.context_build_info.get("context_log_id"),
            context_tokens=workflow_response.context_build_info.get("estimated_tokens"),
            citation_count=len(workflow_response.citations or []),
            status=AnswerStatus.REFUSED.value,
        )

        self.answer_repository.update_answer_log(
            answer_log_id=answer_log_id,
            answer=answer,
            status=AnswerStatus.REFUSED.value,
            latency_ms=latency_ms,
            error_message=None,
        )

        return answer_log_id

    def _sse(
        self,
        event: str,
        data: dict,
    ) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"