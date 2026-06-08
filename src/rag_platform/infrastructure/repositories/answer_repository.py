from sqlalchemy import text

from src.rag_platform.infrastructure.mysql import create_mysql_engine


class AnswerRepository:
    """
    答案生成日志仓储。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def create_answer_log(
        self,
        trace_id: str,
        session_id: str | None,
        question: str,
        rewritten_question: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        context_log_id: int | None,
        context_tokens: int | None,
        citation_count: int,
        status: str,
    ) -> int:
        sql = text("""
            INSERT INTO rag_answer_log (
                trace_id,
                session_id,
                question,
                rewritten_question,
                model,
                temperature,
                max_tokens,
                context_log_id,
                context_tokens,
                citation_count,
                status
            ) VALUES (
                :trace_id,
                :session_id,
                :question,
                :rewritten_question,
                :model,
                :temperature,
                :max_tokens,
                :context_log_id,
                :context_tokens,
                :citation_count,
                :status
            )
        """)

        with self.engine.begin() as conn:
            result = conn.execute(sql, {
                "trace_id": trace_id,
                "session_id": session_id,
                "question": question,
                "rewritten_question": rewritten_question,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "context_log_id": context_log_id,
                "context_tokens": context_tokens,
                "citation_count": citation_count,
                "status": status,
            })
            return int(result.lastrowid)

    def update_answer_log(
        self,
        answer_log_id: int,
        answer: str | None,
        status: str,
        latency_ms: int | None,
        error_message: str | None = None,
    ) -> None:
        sql = text("""
            UPDATE rag_answer_log
            SET
                answer = :answer,
                status = :status,
                latency_ms = :latency_ms,
                error_message = :error_message,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :answer_log_id
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "answer_log_id": answer_log_id,
                "answer": answer,
                "status": status,
                "latency_ms": latency_ms,
                "error_message": error_message,
            })

    def save_answer_citations(
        self,
        answer_log_id: int,
        trace_id: str,
        citations: list[dict],
    ) -> None:
        sql = text("""
            INSERT INTO rag_answer_citation_log (
                answer_log_id,
                trace_id,
                citation_id,
                chunk_id,
                doc_id,
                title,
                title_path,
                source_section,
                chunk_type
            ) VALUES (
                :answer_log_id,
                :trace_id,
                :citation_id,
                :chunk_id,
                :doc_id,
                :title,
                :title_path,
                :source_section,
                :chunk_type
            )
        """)

        with self.engine.begin() as conn:
            for citation in citations:
                conn.execute(sql, {
                    "answer_log_id": answer_log_id,
                    "trace_id": trace_id,
                    "citation_id": citation.get("citation_id"),
                    "chunk_id": citation.get("chunk_id"),
                    "doc_id": citation.get("doc_id"),
                    "title": citation.get("title"),
                    "title_path": citation.get("title_path"),
                    "source_section": citation.get("source_section"),
                    "chunk_type": citation.get("chunk_type"),
                })