from pydantic import ValidationError

from src.rag_platform.infrastructure.deepseek import DeepSeekClient
from src.rag_platform.rag.query_understanding.query_prompt import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_ANALYSIS_USER_PROMPT_TEMPLATE,
)
from src.rag_platform.schemas.query_analysis import QueryAnalysisResult


class LLMQueryAnalyzer:
    """
    LLM Query 分析器。

    职责：
    1. 调用 DeepSeek；
    2. 获取 JSON；
    3. 用 Pydantic 校验成 QueryAnalysisResult。
    """

    def __init__(self) -> None:
        self.client = DeepSeekClient()

    async def analyze(
        self,
        query: str,
        business_domain: str | None,
    ) -> QueryAnalysisResult:
        """
        使用 LLM 分析 query。
        """

        user_prompt = QUERY_ANALYSIS_USER_PROMPT_TEMPLATE.format(
            query=query,
            business_domain=business_domain or "未知",
        )

        raw_json = await self.client.chat_json(
            system_prompt=QUERY_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            return QueryAnalysisResult(
                original_query=query,
                rewritten_query=raw_json.get("rewritten_query") or query,
                expanded_queries=raw_json.get("expanded_queries") or [query],
                target_doc_types=raw_json.get("target_doc_types") or [],
                retrieval_mode=raw_json.get("retrieval_mode") or "hybrid",
                business_domain=raw_json.get("business_domain") or business_domain,
                confidence=float(raw_json.get("confidence") or 0.0),
                reason=raw_json.get("reason") or "",
                need_clarification=bool(raw_json.get("need_clarification") or False),
                clarification_question=raw_json.get("clarification_question"),
                use_llm=True,
                fallback_used=False,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise ValueError(f"LLM QueryAnalysisResult 校验失败: {raw_json}") from exc