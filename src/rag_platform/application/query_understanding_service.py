from uuid import uuid4

from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.repositories.query_analysis_repository import QueryAnalysisRepository
from src.rag_platform.rag.query_understanding.llm_query_analyzer import LLMQueryAnalyzer
from src.rag_platform.rag.query_understanding.rule_based_analyzer import RuleBasedQueryAnalyzer
from src.rag_platform.schemas.query_analysis import (
    QueryAnalysisRequest,
    QueryAnalysisResponse,
    QueryAnalysisResult,
)


class QueryUnderstandingService:
    """
    Query 理解应用服务。

    策略：
    1. 先规则分析，得到稳定兜底结果；
    2. 如果启用 LLM，则调用 LLM 增强；
    3. 如果 LLM 失败或置信度低，则使用规则结果；
    4. 保存分析日志。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.rule_analyzer = RuleBasedQueryAnalyzer()
        self.repository = QueryAnalysisRepository()

    async def analyze(
        self,
        request: QueryAnalysisRequest,
    ) -> QueryAnalysisResponse:
        trace_id = uuid4().hex

        rule_result = self.rule_analyzer.analyze(
            query=request.query,
            business_domain=request.business_domain,
        )

        final_result = rule_result

        if self.settings.query_analysis_use_llm:
            try:
                llm_analyzer = LLMQueryAnalyzer()

                llm_result = await llm_analyzer.analyze(
                    query=request.query,
                    business_domain=request.business_domain,
                )

                if llm_result.confidence >= self.settings.query_analysis_min_confidence:
                    final_result = llm_result
                else:
                    final_result = self._mark_fallback(
                        rule_result,
                        reason=f"LLM置信度过低，使用规则兜底。LLM reason={llm_result.reason}",
                    )

            except Exception as exc:
                final_result = self._mark_fallback(
                    rule_result,
                    reason=f"LLM分析失败，使用规则兜底: {exc}",
                )

        self.repository.save_analysis_log(
            trace_id=trace_id,
            session_id=request.session_id,
            result=final_result,
        )

        return QueryAnalysisResponse(
            trace_id=trace_id,
            result=final_result,
        )

    def _mark_fallback(
        self,
        result: QueryAnalysisResult,
        reason: str,
    ) -> QueryAnalysisResult:
        """
        标记当前结果来自兜底。
        """

        result.use_llm = False
        result.fallback_used = True
        result.reason = reason

        return result