import time

from sqlalchemy import text

from src.rag_platform.domain.rerank import RerankResultItem
from src.rag_platform.infrastructure.mysql import create_mysql_engine


class RerankRepository:
    """
    Rerank 日志仓储。

    保存 rerank 调用日志和候选明细，方便排查：
    1. rerank 前后排序变化；
    2. qwen3-rerank 分数；
    3. fallback 是否触发。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def create_rerank_log(
        self,
        trace_id: str,
        query_text: str,
        provider: str,
        model: str,
        candidate_count: int,
        top_n: int,
        success_count: int,
        status: str,
        fail_open: bool,
        latency_ms: int | None,
        error_message: str | None = None,
    ) -> int:
        sql = text("""
            INSERT INTO rag_rerank_log (
                trace_id,
                query_text,
                provider,
                model,
                candidate_count,
                top_n,
                success_count,
                status,
                fail_open,
                error_message,
                latency_ms
            ) VALUES (
                :trace_id,
                :query_text,
                :provider,
                :model,
                :candidate_count,
                :top_n,
                :success_count,
                :status,
                :fail_open,
                :error_message,
                :latency_ms
            )
        """)

        params = {
            "trace_id": trace_id,
            "query_text": query_text,
            "provider": provider,
            "model": model,
            "candidate_count": candidate_count,
            "top_n": top_n,
            "success_count": success_count,
            "status": status,
            "fail_open": 1 if fail_open else 0,
            "error_message": error_message,
            "latency_ms": latency_ms,
        }

        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
            return int(result.lastrowid)

    def save_rerank_items(
        self,
        rerank_log_id: int,
        trace_id: str,
        items: list[RerankResultItem],
    ) -> None:
        sql = text("""
            INSERT INTO rag_rerank_item_log (
                rerank_log_id,
                trace_id,
                chunk_id,
                before_rank,
                after_rank,
                before_score,
                rerank_score,
                source,
                title,
                title_path
            ) VALUES (
                :rerank_log_id,
                :trace_id,
                :chunk_id,
                :before_rank,
                :after_rank,
                :before_score,
                :rerank_score,
                :source,
                :title,
                :title_path
            )
        """)

        with self.engine.begin() as conn:
            for item in items:
                conn.execute(sql, {
                    "rerank_log_id": rerank_log_id,
                    "trace_id": trace_id,
                    "chunk_id": item.chunk_id,
                    "before_rank": item.metadata.get("before_rank"),
                    "after_rank": item.after_rank,
                    "before_score": item.metadata.get("before_score"),
                    "rerank_score": item.relevance_score,
                    "source": item.metadata.get("source"),
                    "title": item.metadata.get("title"),
                    "title_path": item.metadata.get("title_path"),
                })