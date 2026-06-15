import json

from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.deepseek import DeepSeekClient
from src.rag_platform.rag.adaptive.models import QueryRewriteResult
from src.rag_platform.rag.adaptive.quality_features import (
    extract_exact_terms,
)
from src.rag_platform.rag.adaptive.query_rewrite_prompt import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
)


class QueryRewriter:
    def __init__(
        self,
        *,
        settings=None,
        client=None,
        client_factory=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client
        self.client_factory = client_factory or DeepSeekClient

    async def rewrite(
        self,
        *,
        original_question: str,
        current_queries: list[str],
        quality_reasons: list[str],
        candidate_documents: list[dict],
    ) -> QueryRewriteResult:
        client = self.client or self.client_factory()
        owns_client = self.client is None
        try:
            prompt = QUERY_REWRITE_USER_PROMPT.format(
                original_question=original_question,
                current_queries=json.dumps(
                    current_queries[:5],
                    ensure_ascii=False,
                ),
                quality_reasons=json.dumps(
                    quality_reasons,
                    ensure_ascii=False,
                ),
                candidate_summaries=json.dumps(
                    self._candidate_summaries(candidate_documents),
                    ensure_ascii=False,
                ),
            )
            last_error: Exception | None = None
            for _ in range(
                self.settings.adaptive_rewrite_max_attempts
            ):
                try:
                    raw = await client.chat_json(
                        system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                        user_prompt=prompt,
                        model=self.settings.adaptive_rewrite_model,
                        temperature=0,
                        max_tokens=512,
                    )
                    return self._parse_model_result(raw)
                except Exception as exc:
                    last_error = exc
            return self._fallback(
                original_question=original_question,
                current_queries=current_queries,
                quality_reasons=quality_reasons,
                error=last_error,
            )
        finally:
            close = getattr(client, "aclose", None)
            if owns_client and close is not None:
                await close()

    @staticmethod
    def _parse_model_result(raw: dict) -> QueryRewriteResult:
        rewritten = str(raw.get("rewritten_query") or "").strip()
        expanded = raw.get("expanded_queries")
        if not rewritten or not isinstance(expanded, list):
            raise ValueError("Query改写模型返回字段不完整")
        queries = list(
            dict.fromkeys(
                str(value).strip()
                for value in expanded
                if str(value).strip()
            )
        )
        return QueryRewriteResult(
            rewritten_query=rewritten,
            expanded_queries=[
                query
                for query in queries
                if query != rewritten
            ][:2],
            reason=str(raw.get("reason") or "").strip(),
            fallback_used=False,
        )

    @staticmethod
    def _candidate_summaries(
        documents: list[dict],
    ) -> list[dict[str, str]]:
        summaries = []
        for document in documents[:5]:
            metadata = document.get("metadata") or {}
            title = (
                document.get("title")
                or metadata.get("title")
                or metadata.get("title_path")
                or ""
            )
            content = str(
                document.get("page_content") or ""
            ).strip()
            summaries.append(
                {
                    "title": str(title)[:80],
                    "summary": content[:160],
                }
            )
        return summaries

    @staticmethod
    def _fallback(
        *,
        original_question: str,
        current_queries: list[str],
        quality_reasons: list[str],
        error: Exception | None,
    ) -> QueryRewriteResult:
        exact_terms = extract_exact_terms(original_question)
        comparison = any(
            value in original_question
            for value in ("新旧", "新版", "旧版", "版本", "区别", "不同")
        )
        rewritten = " ".join(
            [*exact_terms, original_question]
        ).strip()
        expanded: list[str] = []
        if comparison:
            expanded.extend(
                [
                    f"{original_question} 旧版 规则",
                    f"{original_question} 新版 生效规则",
                ]
            )
        else:
            expanded.extend(current_queries[:2])
        reason_parts = ["Query改写模型不可用，使用确定性兜底"]
        if quality_reasons:
            reason_parts.append("；".join(quality_reasons))
        if error is not None:
            reason_parts.append(type(error).__name__)
        return QueryRewriteResult(
            rewritten_query=rewritten,
            expanded_queries=[
                query
                for query in list(dict.fromkeys(expanded))
                if query and query != rewritten
            ][:2],
            reason="；".join(reason_parts),
            fallback_used=True,
        )
