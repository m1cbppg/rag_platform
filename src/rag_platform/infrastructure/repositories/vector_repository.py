import json
from typing import Any

from sqlalchemy import text

from src.rag_platform.infrastructure.mysql import create_mysql_engine


class VectorRepository:
    """
    向量相关仓储层。

    Repository 只负责 MySQL 读写，不负责 Milvus 的具体操作。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def upsert_collection_state(
        self,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_output_type: str,
        vector_field: str,
        metric_type: str,
        index_type: str,
        index_params: dict[str, Any],
        status: str,
        error_message: str | None = None,
    ) -> None:
        """
        保存或更新 Milvus Collection 状态。

        MySQL 里 collection_name 是唯一键。
        如果 collection_name 已存在，则执行更新。
        """

        sql = text("""
            INSERT INTO rag_vector_collection_state (
                collection_name,
                embedding_model,
                embedding_dimension,
                embedding_output_type,
                vector_field,
                metric_type,
                index_type,
                index_params_json,
                status,
                error_message
            ) VALUES (
                :collection_name,
                :embedding_model,
                :embedding_dimension,
                :embedding_output_type,
                :vector_field,
                :metric_type,
                :index_type,
                CAST(:index_params_json AS JSON),
                :status,
                :error_message
            )
            ON DUPLICATE KEY UPDATE
                embedding_model = VALUES(embedding_model),
                embedding_dimension = VALUES(embedding_dimension),
                embedding_output_type = VALUES(embedding_output_type),
                vector_field = VALUES(vector_field),
                metric_type = VALUES(metric_type),
                index_type = VALUES(index_type),
                index_params_json = VALUES(index_params_json),
                status = VALUES(status),
                error_message = VALUES(error_message),
                updated_at = CURRENT_TIMESTAMP
        """)

        params = {
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_output_type": embedding_output_type,
            "vector_field": vector_field,
            "metric_type": metric_type,
            "index_type": index_type,
            "index_params_json": json.dumps(index_params, ensure_ascii=False),
            "status": status,
            "error_message": error_message,
        }

        with self.engine.begin() as conn:
            conn.execute(sql, params)

    def get_collection_state(self, collection_name: str) -> dict | None:
        """
        查询 collection 状态。
        """

        sql = text("""
            SELECT
                id,
                collection_name,
                embedding_model,
                embedding_dimension,
                embedding_output_type,
                vector_field,
                metric_type,
                index_type,
                index_params_json,
                status,
                error_message,
                created_at,
                updated_at
            FROM rag_vector_collection_state
            WHERE collection_name = :collection_name
            LIMIT 1
        """)

        with self.engine.begin() as conn:
            row = conn.execute(
                sql,
                {"collection_name": collection_name},
            ).mappings().first()

        if row is None:
            return None

        result = dict(row)

        if isinstance(result.get("index_params_json"), str):
            result["index_params_json"] = json.loads(result["index_params_json"])

        return result

    def list_active_chunks_for_embedding(
            self,
            doc_id: int | None,
            limit: int,
    ) -> list[dict]:
        """
        查询需要创建 embedding task 的 chunk。

        只处理 ACTIVE 状态的 chunk。
        如果传入 doc_id，则只处理指定文档。
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

    def upsert_embedding_task(
            self,
            chunk_id: int,
            doc_id: int,
            embedding_model: str,
            embedding_dimension: int,
            embedding_output_type: str,
            embedding_text_hash: str,
            milvus_collection: str,
    ) -> None:
        """
        创建或更新 embedding task。

        这里使用 ON DUPLICATE KEY UPDATE。

        如果同一个 chunk + 模型 + 维度 + output_type 已存在：
        - 如果 hash 变化，则重新设为 PENDING；
        - 如果 hash 没变，也更新 updated_at，但后面执行时可以跳过。
        """

        sql = text("""
            INSERT INTO rag_embedding_task (
                chunk_id,
                doc_id,
                embedding_model,
                embedding_dimension,
                embedding_output_type,
                embedding_text_hash,
                milvus_collection,
                status
            ) VALUES (
                :chunk_id,
                :doc_id,
                :embedding_model,
                :embedding_dimension,
                :embedding_output_type,
                :embedding_text_hash,
                :milvus_collection,
                'PENDING'
            )
            ON DUPLICATE KEY UPDATE
                status = CASE
                    WHEN embedding_text_hash <> VALUES(embedding_text_hash)
                      OR status = 'PROCESSING'
                    THEN 'PENDING'
                    ELSE status
                END,
                embedding_text_hash = VALUES(embedding_text_hash),
                milvus_collection = VALUES(milvus_collection),
                updated_at = CURRENT_TIMESTAMP
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "embedding_output_type": embedding_output_type,
                "embedding_text_hash": embedding_text_hash,
                "milvus_collection": milvus_collection,
            })

    def list_pending_embedding_tasks(
        self,
        limit: int,
        doc_id: int | None = None,
    ) -> list[dict]:
        """
        查询待执行的 embedding 任务。

        只查 PENDING / FAILED。
        FAILED 也查出来，是为了支持失败重试。
        """

        doc_filter = "AND t.doc_id = :doc_id" if doc_id is not None else ""
        sql = text(f"""
            SELECT
                t.id AS task_id,
                t.chunk_id,
                t.doc_id,
                t.embedding_model,
                t.embedding_dimension,
                t.embedding_output_type,
                t.embedding_text_hash,
                t.milvus_collection,
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
            FROM rag_embedding_task t
            JOIN rag_chunk c ON t.chunk_id = c.id
            WHERE t.status IN ('PENDING', 'FAILED')
              AND c.status = 'ACTIVE'
              {doc_filter}
            ORDER BY t.updated_at ASC
            LIMIT :limit
        """)

        params = {"limit": limit}
        if doc_id is not None:
            params["doc_id"] = doc_id
        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [dict(row) for row in rows]

    def get_embedding_task_summary(
        self,
        doc_id: int,
        embedding_model: str,
        embedding_dimension: int,
        embedding_output_type: str,
    ) -> dict[str, int]:
        sql = text(
            """
            SELECT t.status, COUNT(*) AS count
            FROM rag_embedding_task t
            JOIN rag_chunk c ON c.id = t.chunk_id
            WHERE t.doc_id = :doc_id
              AND t.embedding_model = :embedding_model
              AND t.embedding_dimension = :embedding_dimension
              AND t.embedding_output_type = :embedding_output_type
              AND c.status = 'ACTIVE'
            GROUP BY t.status
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(
                sql,
                {
                    "doc_id": doc_id,
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "embedding_output_type": embedding_output_type,
                },
            ).mappings().all()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def update_embedding_task_status(
            self,
            task_id: int,
            status: str,
            milvus_pk: int | None = None,
            error_message: str | None = None,
            increase_retry: bool = False,
    ) -> None:
        """
        更新 embedding task 状态。

        increase_retry=True 时，retry_count + 1。
        """

        retry_sql = "retry_count = retry_count + 1," if increase_retry else ""

        sql = text(f"""
            UPDATE rag_embedding_task
            SET
                status = :status,
                milvus_pk = :milvus_pk,
                {retry_sql}
                error_message = :error_message,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :task_id
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "task_id": task_id,
                "status": status,
                "milvus_pk": milvus_pk,
                "error_message": error_message,
            })

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> dict[int, dict]:
        """
        根据 chunk_id 批量查询 MySQL chunk 内容。

        Milvus 检索只返回 chunk_id 和 score。
        最终还要回 MySQL 查完整内容。
        """

        if not chunk_ids:
            return {}

        placeholders = ", ".join([f":id_{index}" for index in range(len(chunk_ids))])

        sql = text(f"""
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
            WHERE id IN ({placeholders})
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
