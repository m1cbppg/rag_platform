import hashlib
from typing import Any

from sqlalchemy import text

from src.rag_platform.infrastructure.mysql import create_mysql_engine


class SearchRepository:
    """
    搜索索引仓储层。

    负责：
    1. 查询 chunk；
    2. 创建 ES index task；
    3. 查询待执行 task；
    4. 更新 task 状态；
    5. 保存 index 状态。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def upsert_index_state(
        self,
        search_engine: str,
        index_name: str,
        analyzer: str,
        search_analyzer: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        sql = text("""
            INSERT INTO rag_search_index_state (
                search_engine,
                index_name,
                analyzer,
                search_analyzer,
                status,
                error_message
            ) VALUES (
                :search_engine,
                :index_name,
                :analyzer,
                :search_analyzer,
                :status,
                :error_message
            )
            ON DUPLICATE KEY UPDATE
                analyzer = VALUES(analyzer),
                search_analyzer = VALUES(search_analyzer),
                status = VALUES(status),
                error_message = VALUES(error_message),
                updated_at = CURRENT_TIMESTAMP
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "search_engine": search_engine,
                "index_name": index_name,
                "analyzer": analyzer,
                "search_analyzer": search_analyzer,
                "status": status,
                "error_message": error_message,
            })

    def list_active_chunks_for_indexing(
        self,
        doc_id: int | None,
        limit: int,
    ) -> list[dict]:
        """
        查询需要写入 ES 的 active chunk。
        """

        if doc_id is None:
            sql = text("""
                SELECT
                    id,
                    chunk_code,
                    doc_id,
                    chunk_type,
                    title,
                    title_path,
                    content,
                    keywords,
                    tags,
                    business_domain,
                    version,
                    source_section,
                    status
                FROM rag_chunk
                WHERE status = 'ACTIVE'
                ORDER BY id ASC
                LIMIT :limit
            """)
            params = {"limit": limit}
        else:
            sql = text("""
                SELECT
                    id,
                    chunk_code,
                    doc_id,
                    chunk_type,
                    title,
                    title_path,
                    content,
                    keywords,
                    tags,
                    business_domain,
                    version,
                    source_section,
                    status
                FROM rag_chunk
                WHERE status = 'ACTIVE'
                  AND doc_id = :doc_id
                ORDER BY id ASC
                LIMIT :limit
            """)
            params = {"doc_id": doc_id, "limit": limit}

        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [dict(row) for row in rows]

    def build_index_text(self, chunk: dict[str, Any]) -> str:
        """
        构造 ES 索引文本。

        注意：
        ES 文档会拆成 title/content/keywords 等字段。
        这里的 index_text 主要用于计算 hash。
        """

        parts = [
            str(chunk.get("chunk_type") or ""),
            str(chunk.get("title") or ""),
            str(chunk.get("title_path") or ""),
            str(chunk.get("content") or ""),
            str(chunk.get("keywords") or ""),
            str(chunk.get("tags") or ""),
            str(chunk.get("source_section") or ""),
        ]

        return "\n".join(parts)

    def hash_index_text(self, text_value: str) -> str:
        return hashlib.sha256(text_value.encode("utf-8")).hexdigest()

    def upsert_keyword_index_task(
        self,
        chunk_id: int,
        doc_id: int,
        index_name: str,
        index_text_hash: str,
    ) -> None:
        sql = text("""
            INSERT INTO rag_keyword_index_task (
                chunk_id,
                doc_id,
                search_engine,
                index_name,
                index_text_hash,
                status
            ) VALUES (
                :chunk_id,
                :doc_id,
                'elasticsearch',
                :index_name,
                :index_text_hash,
                'PENDING'
            )
            ON DUPLICATE KEY UPDATE
                index_text_hash = VALUES(index_text_hash),
                status = CASE
                    WHEN index_text_hash <> VALUES(index_text_hash)
                    THEN 'PENDING'
                    ELSE status
                END,
                updated_at = CURRENT_TIMESTAMP
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "index_name": index_name,
                "index_text_hash": index_text_hash,
            })

    def list_pending_keyword_index_tasks(self, limit: int) -> list[dict]:
        sql = text("""
            SELECT
                t.id AS task_id,
                t.chunk_id,
                t.doc_id,
                t.index_name,
                t.retry_count,

                c.chunk_code,
                c.chunk_type,
                c.title,
                c.title_path,
                c.content,
                c.keywords,
                c.tags,
                c.business_domain,
                c.version,
                c.source_section,
                c.status AS chunk_status
            FROM rag_keyword_index_task t
            JOIN rag_chunk c ON t.chunk_id = c.id
            WHERE t.status IN ('PENDING', 'FAILED')
              AND c.status = 'ACTIVE'
            ORDER BY t.updated_at ASC
            LIMIT :limit
        """)

        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"limit": limit}).mappings().all()

        return [dict(row) for row in rows]

    def update_keyword_index_task_status(
        self,
        task_id: int,
        status: str,
        error_message: str | None = None,
        increase_retry: bool = False,
    ) -> None:
        retry_sql = "retry_count = retry_count + 1," if increase_retry else ""

        sql = text(f"""
            UPDATE rag_keyword_index_task
            SET
                status = :status,
                {retry_sql}
                error_message = :error_message,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :task_id
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "task_id": task_id,
                "status": status,
                "error_message": error_message,
            })