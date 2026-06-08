from sqlalchemy import text

from src.rag_platform.domain.context import Citation
from src.rag_platform.infrastructure.mysql import create_mysql_engine


class ContextRepository:
    """
    Context 相关仓储层。

    注意：
    这里不负责构建 context，只负责 MySQL 查询和日志保存。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> dict[int, dict]:
        """
        根据 chunk_id 批量查询 chunk。
        """

        if not chunk_ids:
            return {}

        placeholders = ", ".join(
            [f":id_{index}" for index in range(len(chunk_ids))]
        )

        sql = text(f"""
            SELECT
                id,
                chunk_code,
                doc_id,
                parent_chunk_id,
                chunk_type,
                title,
                title_path,
                content,
                summary,
                keywords,
                tags,
                business_domain,
                version,
                source_doc_title,
                source_page,
                source_section,
                token_count,
                sort_order,
                status
            FROM rag_chunk
            WHERE id IN ({placeholders})
              AND status = 'ACTIVE'
        """)

        params = {
            f"id_{index}": chunk_id
            for index, chunk_id in enumerate(chunk_ids)
        }

        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return {
            int(row["id"]): dict(row)
            for row in rows
        }

    def get_related_chunk_ids(
        self,
        chunk_id: int,
        relation_types: list[str],
        limit: int,
    ) -> list[tuple[int, str]]:
        """
        查询某个 chunk 的相关 chunk。

        返回：
            [(related_chunk_id, relation_type), ...]

        同时查 from_chunk_id 和 to_chunk_id 两个方向。
        """

        if not relation_types:
            return []

        placeholders = ", ".join(
            [f":type_{index}" for index in range(len(relation_types))]
        )

        sql = text(f"""
            SELECT
                CASE
                    WHEN from_chunk_id = :chunk_id THEN to_chunk_id
                    ELSE from_chunk_id
                END AS related_chunk_id,
                relation_type
            FROM rag_chunk_relation
            WHERE (from_chunk_id = :chunk_id OR to_chunk_id = :chunk_id)
              AND relation_type IN ({placeholders})
            ORDER BY sort_order ASC
            LIMIT :limit
        """)

        params = {
            "chunk_id": chunk_id,
            "limit": limit,
        }

        for index, relation_type in enumerate(relation_types):
            params[f"type_{index}"] = relation_type

        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [
            (int(row["related_chunk_id"]), str(row["relation_type"]))
            for row in rows
        ]

    def create_context_log(
        self,
        trace_id: str,
        query_text: str,
        input_document_count: int,
        final_chunk_count: int,
        max_tokens: int,
        estimated_tokens: int,
        expand_parent: bool,
        expand_previous_next: bool,
        expand_same_section: bool,
        status: str,
        error_message: str | None = None,
    ) -> int:
        """
        保存 context 构建日志主表。
        """

        sql = text("""
            INSERT INTO rag_context_build_log (
                trace_id,
                query_text,
                input_document_count,
                final_chunk_count,
                max_tokens,
                estimated_tokens,
                expand_parent,
                expand_previous_next,
                expand_same_section,
                status,
                error_message
            ) VALUES (
                :trace_id,
                :query_text,
                :input_document_count,
                :final_chunk_count,
                :max_tokens,
                :estimated_tokens,
                :expand_parent,
                :expand_previous_next,
                :expand_same_section,
                :status,
                :error_message
            )
        """)

        params = {
            "trace_id": trace_id,
            "query_text": query_text,
            "input_document_count": input_document_count,
            "final_chunk_count": final_chunk_count,
            "max_tokens": max_tokens,
            "estimated_tokens": estimated_tokens,
            "expand_parent": 1 if expand_parent else 0,
            "expand_previous_next": 1 if expand_previous_next else 0,
            "expand_same_section": 1 if expand_same_section else 0,
            "status": status,
            "error_message": error_message,
        }

        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
            return int(result.lastrowid)

    def save_citation_logs(
        self,
        context_log_id: int,
        trace_id: str,
        citations: list[Citation],
    ) -> None:
        """
        保存 citation 明细。
        """

        sql = text("""
            INSERT INTO rag_context_citation_log (
                context_log_id,
                trace_id,
                citation_id,
                chunk_id,
                doc_id,
                title,
                title_path,
                source_section,
                chunk_type,
                expansion_type,
                sort_order
            ) VALUES (
                :context_log_id,
                :trace_id,
                :citation_id,
                :chunk_id,
                :doc_id,
                :title,
                :title_path,
                :source_section,
                :chunk_type,
                :expansion_type,
                :sort_order
            )
        """)

        with self.engine.begin() as conn:
            for citation in citations:
                conn.execute(sql, {
                    "context_log_id": context_log_id,
                    "trace_id": trace_id,
                    "citation_id": citation.citation_id,
                    "chunk_id": citation.chunk_id,
                    "doc_id": citation.doc_id,
                    "title": citation.title,
                    "title_path": citation.title_path,
                    "source_section": citation.source_section,
                    "chunk_type": citation.chunk_type,
                    "expansion_type": citation.expansion_type,
                    "sort_order": citation.sort_order,
                })