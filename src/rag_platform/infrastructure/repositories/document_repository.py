import json
import logging
from typing import Any

from sqlalchemy import text

from src.rag_platform.infrastructure.mysql import create_mysql_engine

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    文档仓储层。

    Repository 的职责：
    1. 保存文档主表；
    2. 保存解析结果；
    3. 保存质量校验结果；
    4. 更新文档状态。

    它只关心数据库，不关心文档怎么解析。
    """

    def __init__(self) -> None:
        self.engine = create_mysql_engine()

    def create_document(
            self,
            doc_code: str,
            title: str,
            doc_type: str,
            file_name: str,
            file_path: str,
            file_ext: str,
            file_sha256: str,
            business_domain: str | None,
            version: str | None,
            created_by: str | None,
    ) -> int:
        sql = text("""
            INSERT INTO rag_document (
                doc_code, title, doc_type, file_name, file_path, file_ext, file_sha256,
                business_domain, version, status, created_by
            ) VALUES (
                :doc_code, :title, :doc_type, :file_name, :file_path, :file_ext, :file_sha256,
                :business_domain, :version, 'UPLOADED', :created_by
            )
        """)

        params = {
            "doc_code": doc_code,
            "title": title,
            "doc_type": doc_type,
            "file_name": file_name,
            "file_path": file_path,
            "file_ext": file_ext,
            "file_sha256": file_sha256,
            "business_domain": business_domain,
            "version": version,
            "created_by": created_by,
        }

        logger.info(
            "documents.repository.create_document.start doc_code=%s title=%s "
            "doc_type=%s file_name=%s file_ext=%s",
            doc_code,
            title,
            doc_type,
            file_name,
            file_ext,
        )
        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
            doc_id = int(result.lastrowid)

        logger.info(
            "documents.repository.create_document.completed doc_id=%s doc_code=%s",
            doc_id,
            doc_code,
        )
        return doc_id

    def update_status(self, doc_id: int, status: str) -> None:
        sql = text("""
            UPDATE rag_document
            SET status = :status
            WHERE id = :doc_id
        """)

        logger.info(
            "documents.repository.update_status.start doc_id=%s status=%s",
            doc_id,
            status,
        )
        with self.engine.begin() as conn:
            conn.execute(sql, {"doc_id": doc_id, "status": status})
        logger.info(
            "documents.repository.update_status.completed doc_id=%s status=%s",
            doc_id,
            status,
        )

    def save_parse_result(
        self,
        doc_id: int,
        parser_type: str,
        raw_content: str,
        clean_content: str,
        structure: dict[str, Any],
        parse_status: str,
        error_message: str | None = None,
    ) -> None:
        sql = text("""
            INSERT INTO rag_document_parse (
                doc_id, parser_type, raw_content, clean_content,
                structure_json, parse_status, error_message
            ) VALUES (
                :doc_id, :parser_type, :raw_content, :clean_content,
                CAST(:structure_json AS JSON), :parse_status, :error_message
            )
        """)

        params = {
            "doc_id": doc_id,
            "parser_type": parser_type,
            "raw_content": raw_content,
            "clean_content": clean_content,
            "structure_json": json.dumps(structure, ensure_ascii=False),
            "parse_status": parse_status,
            "error_message": error_message,
        }

        logger.info(
            "documents.repository.save_parse_result.start doc_id=%s parser_type=%s "
            "parse_status=%s raw_content_length=%s clean_content_length=%s "
            "structure_keys=%s has_error=%s",
            doc_id,
            parser_type,
            parse_status,
            len(raw_content),
            len(clean_content),
            list(structure.keys()),
            error_message is not None,
        )
        with self.engine.begin() as conn:
            conn.execute(sql, params)
        logger.info(
            "documents.repository.save_parse_result.completed doc_id=%s parse_status=%s",
            doc_id,
            parse_status,
        )

    def save_quality_results(self, doc_id: int, results: list[dict]) -> None:
        sql = text("""
            INSERT INTO rag_document_quality (
                doc_id, check_item, check_result, message
            ) VALUES (
                :doc_id, :check_item, :check_result, :message
            )
        """)

        logger.info(
            "documents.repository.save_quality_results.start doc_id=%s result_count=%s",
            doc_id,
            len(results),
        )
        with self.engine.begin() as conn:
            for item in results:
                logger.info(
                    "documents.repository.save_quality_result.item doc_id=%s "
                    "check_item=%s check_result=%s",
                    doc_id,
                    item["check_item"],
                    item["check_result"],
                )
                conn.execute(sql, {
                    "doc_id": doc_id,
                    "check_item": item["check_item"],
                    "check_result": item["check_result"],
                    "message": item["message"],
                })
        logger.info(
            "documents.repository.save_quality_results.completed doc_id=%s result_count=%s",
            doc_id,
            len(results),
        )

    def find_by_file_sha256(self, file_sha256: str) -> dict | None:
        """
        根据文件 SHA256 查询是否已经上传过。

        返回：
        - 如果存在，返回文档记录 dict；
        - 如果不存在，返回 None。
        """

        sql = text("""
            SELECT
                id,
                doc_code,
                title,
                doc_type,
                file_name,
                file_path,
                file_ext,
                file_sha256,
                business_domain,
                version,
                status,
                created_at,
                updated_at
            FROM rag_document
            WHERE file_sha256 = :file_sha256
            LIMIT 1
        """)

        with self.engine.begin() as conn:
            row = conn.execute(sql, {"file_sha256": file_sha256}).mappings().first()

        if row is None:
            return None

        return dict(row)


    def get_cleaned_document_parse(self, doc_id: int) -> dict | None:
        """
        查询已经清洗完成的文档解析结果。

        模块 3 会读取：
        1. rag_document 文档基础信息；
        2. rag_document_parse.structure_json；
        3. rag_document_parse.clean_content。

        只有 CLEANED 状态的文档才能切 chunk。
        """

        sql = text("""
            SELECT
                d.id AS doc_id,
                d.title AS doc_title,
                d.doc_type AS doc_type,
                d.business_domain AS business_domain,
                d.version AS version,
                d.status AS doc_status,
                p.clean_content AS clean_content,
                p.structure_json AS structure_json
            FROM rag_document d
            JOIN rag_document_parse p ON d.id = p.doc_id
            WHERE d.id = :doc_id
              AND d.status = 'CLEANED'
            ORDER BY p.id DESC
            LIMIT 1
        """)

        with self.engine.begin() as conn:
            row = conn.execute(sql, {"doc_id": doc_id}).mappings().first()

        if row is None:
            return None

        result = dict(row)

        structure_json = result.get("structure_json")

        # PyMySQL 读取 JSON 字段时，有时返回字符串，有时可能已经是 dict。
        # 所以这里做兼容处理。
        if isinstance(structure_json, str):
            result["structure_json"] = json.loads(structure_json)

        return result

    def delete_chunks_by_doc_id(self, doc_id: int) -> None:
        """
        删除某个文档已经生成的 chunk。

        为什么需要？
        1. 支持重复执行 chunk 构建；
        2. 保证 chunk 构建幂等；
        3. 避免同一文档反复生成重复 chunk。

        注意：
        先删 relation，再删 chunk。
        因为 relation 表有外键依赖 chunk。
        """

        delete_relation_sql = text("""
            DELETE r
            FROM rag_chunk_relation r
            JOIN rag_chunk c ON r.from_chunk_id = c.id OR r.to_chunk_id = c.id
            WHERE c.doc_id = :doc_id
        """)

        delete_chunk_sql = text("""
            DELETE FROM rag_chunk
            WHERE doc_id = :doc_id
        """)

        with self.engine.begin() as conn:
            conn.execute(delete_relation_sql, {"doc_id": doc_id})
            conn.execute(delete_chunk_sql, {"doc_id": doc_id})

    def create_chunk(
            self,
            chunk_code: str,
            doc_id: int,
            parent_chunk_id: int | None,
            chunk_type: str,
            title: str | None,
            title_path: str | None,
            content: str,
            summary: str | None,
            keywords: str | None,
            tags: str | None,
            business_domain: str | None,
            version: str | None,
            source_doc_title: str | None,
            source_page: int | None,
            source_section: str | None,
            token_count: int,
            sort_order: int,
    ) -> int:
        """
        新增 chunk，并返回数据库自增 ID。
        """

        sql = text("""
            INSERT INTO rag_chunk (
                chunk_code, doc_id, parent_chunk_id, chunk_type,
                title, title_path, content, summary, keywords, tags,
                business_domain, version, source_doc_title, source_page,
                source_section, token_count, sort_order, status
            ) VALUES (
                :chunk_code, :doc_id, :parent_chunk_id, :chunk_type,
                :title, :title_path, :content, :summary, :keywords, :tags,
                :business_domain, :version, :source_doc_title, :source_page,
                :source_section, :token_count, :sort_order, 'ACTIVE'
            )
        """)

        params = {
            "chunk_code": chunk_code,
            "doc_id": doc_id,
            "parent_chunk_id": parent_chunk_id,
            "chunk_type": chunk_type,
            "title": title,
            "title_path": title_path,
            "content": content,
            "summary": summary,
            "keywords": keywords,
            "tags": tags,
            "business_domain": business_domain,
            "version": version,
            "source_doc_title": source_doc_title,
            "source_page": source_page,
            "source_section": source_section,
            "token_count": token_count,
            "sort_order": sort_order,
        }

        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
            return int(result.lastrowid)

    def create_chunk_relation(
            self,
            from_chunk_id: int,
            to_chunk_id: int,
            relation_type: str,
            sort_order: int,
    ) -> None:
        """
        新增 chunk 关系。
        """

        sql = text("""
            INSERT INTO rag_chunk_relation (
                from_chunk_id, to_chunk_id, relation_type, sort_order
            ) VALUES (
                :from_chunk_id, :to_chunk_id, :relation_type, :sort_order
            )
        """)

        with self.engine.begin() as conn:
            conn.execute(sql, {
                "from_chunk_id": from_chunk_id,
                "to_chunk_id": to_chunk_id,
                "relation_type": relation_type,
                "sort_order": sort_order,
            })

    def list_chunks_by_doc_id(self, doc_id: int) -> list[dict]:
        sql = text(
            """
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
            WHERE doc_id = :doc_id
              AND status = 'ACTIVE'
            ORDER BY sort_order ASC, id ASC
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"doc_id": doc_id}).mappings().all()
        return [dict(row) for row in rows]
