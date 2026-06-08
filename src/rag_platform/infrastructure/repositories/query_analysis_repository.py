import json

from sqlalchemy import text

from src.rag_platform.infrastructure.mysql import create_mysql_engine
from src.rag_platform.schemas.query_analysis import QueryAnalysisResult


class QueryAnalysisRepository:
    """
    Query 分析日志仓储。

    当前只保存分析结果，方便调试路由。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def save_analysis_log(
        self,
        trace_id: str,
        session_id: str | None,
        result: QueryAnalysisResult,
        raw_llm_output: dict | None = None,
    ) -> None:
        """
        保存 Query 分析日志。
        """

        sql = text("""
            INSERT INTO rag_query_analysis_log (
                trace_id,
                session_id,
                original_query,
                rewritten_query,
                expanded_queries_json,
                target_doc_types_json,
                retrieval_mode,
                business_domain,
                confidence,
                use_llm,
                fallback_used,
                reason,
                raw_llm_output
            ) VALUES (
                :trace_id,
                :session_id,
                :original_query,
                :rewritten_query,
                CAST(:expanded_queries_json AS JSON),
                CAST(:target_doc_types_json AS JSON),
                :retrieval_mode,
                :business_domain,
                :confidence,
                :use_llm,
                :fallback_used,
                :reason,
                CAST(:raw_llm_output AS JSON)
            )
        """)

        params = {
            "trace_id": trace_id,
            "session_id": session_id,
            "original_query": result.original_query,
            "rewritten_query": result.rewritten_query,
            "expanded_queries_json": json.dumps(result.expanded_queries, ensure_ascii=False),
            "target_doc_types_json": json.dumps(result.target_doc_types, ensure_ascii=False),
            "retrieval_mode": result.retrieval_mode,
            "business_domain": result.business_domain,
            "confidence": result.confidence,
            "use_llm": 1 if result.use_llm else 0,
            "fallback_used": 1 if result.fallback_used else 0,
            "reason": result.reason,
            "raw_llm_output": json.dumps(raw_llm_output or {}, ensure_ascii=False),
        }

        with self.engine.begin() as conn:
            conn.execute(sql, params)