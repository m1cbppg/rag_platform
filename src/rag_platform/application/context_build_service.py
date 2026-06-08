from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.context import ContextBuildResult
from src.rag_platform.infrastructure.repositories.context_repository import ContextRepository
from src.rag_platform.rag.context.context_builder import ContextBuilder


class ContextBuildService:
    """
    Context 构建应用服务。

    职责：
    1. 调用 ContextBuilder；
    2. 保存 context 构建日志；
    3. 保存 citation 日志；
    4. 返回构建结果。
    """

    def __init__(
        self,
        settings=None,
        builder=None,
        repository=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.builder = builder or ContextBuilder()
        self.repository = repository or ContextRepository()

    def build_context(
        self,
        trace_id: str,
        query_text: str,
        documents: list[dict],
    ) -> tuple[ContextBuildResult, dict]:
        """
        构建上下文，并落日志。

        返回：
            ContextBuildResult
            context_build_info
        """

        try:
            result = self.builder.build(documents)

            context_log_id = self.repository.create_context_log(
                trace_id=trace_id,
                query_text=query_text,
                input_document_count=len(documents),
                final_chunk_count=len(result.used_chunks),
                max_tokens=self.settings.context_max_tokens,
                estimated_tokens=result.estimated_tokens,
                expand_parent=self.settings.context_expand_parent,
                expand_previous_next=self.settings.context_expand_previous_next,
                expand_same_section=self.settings.context_expand_same_section,
                status=result.status.value,
                error_message=None,
            )

            self.repository.save_citation_logs(
                context_log_id=context_log_id,
                trace_id=trace_id,
                citations=result.citations,
            )

            return result, {
                "context_log_id": context_log_id,
                "status": result.status.value,
                "estimated_tokens": result.estimated_tokens,
                "used_chunk_count": len(result.used_chunks),
                "citation_count": len(result.citations),
                "message": result.message,
            }

        except Exception as exc:
            context_log_id = self.repository.create_context_log(
                trace_id=trace_id,
                query_text=query_text,
                input_document_count=len(documents),
                final_chunk_count=0,
                max_tokens=self.settings.context_max_tokens,
                estimated_tokens=0,
                expand_parent=self.settings.context_expand_parent,
                expand_previous_next=self.settings.context_expand_previous_next,
                expand_same_section=self.settings.context_expand_same_section,
                status="FAILED",
                error_message=str(exc),
            )

            raise
