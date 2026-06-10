import hashlib
import json
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.rag_platform.evaluation.models import (
    ActualAction,
    DatasetSplit,
    DatasetStatus,
    EvalCaseResultStatus,
    EvalRunConfig,
    EvalRunStatus,
    EvalCaseStatus,
    EvidenceSpec,
    GeneratedEvalCase,
    JudgeScore,
    MappingStatus,
    RetrievalMetricResult,
    ReviewStatus,
    ReviewedEvalCase,
    SourceDocumentSpec,
)
from src.rag_platform.infrastructure.mysql import create_mysql_engine
from src.rag_platform.rag.retrieval.business_domain import (
    resolve_business_domains,
)


def _enum_value(value: Enum | str) -> str:
    return value.value if isinstance(value, Enum) else value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


class DatasetRepository:
    """RAG 评测数据集与实验结果的 MySQL 仓储。"""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or create_mysql_engine()

    def create_dataset(
        self,
        dataset_code: str,
        name: str,
        version: str,
        domain: str,
        description: str | None = None,
        generator_provider: str | None = None,
        generator_model: str | None = None,
        reviewer_provider: str | None = None,
        reviewer_model: str | None = None,
        generation_config: dict[str, Any] | None = None,
        status: DatasetStatus = DatasetStatus.DRAFT,
    ) -> int:
        sql = text(
            """
            INSERT INTO rag_eval_dataset (
                dataset_code,
                name,
                version,
                domain,
                description,
                generator_provider,
                generator_model,
                reviewer_provider,
                reviewer_model,
                status,
                generation_config_json
            ) VALUES (
                :dataset_code,
                :name,
                :version,
                :domain,
                :description,
                :generator_provider,
                :generator_model,
                :reviewer_provider,
                :reviewer_model,
                :status,
                CAST(:generation_config_json AS JSON)
            )
            """
        )
        params = {
            "dataset_code": dataset_code,
            "name": name,
            "version": version,
            "domain": domain,
            "description": description,
            "generator_provider": generator_provider,
            "generator_model": generator_model,
            "reviewer_provider": reviewer_provider,
            "reviewer_model": reviewer_model,
            "status": _enum_value(status),
            "generation_config_json": _json_dumps(generation_config or {}),
        }

        with self.engine.begin() as connection:
            result = connection.execute(sql, params)
            return int(result.lastrowid)

    def find_dataset(
        self,
        dataset_code: str,
        version: str,
    ) -> dict[str, Any] | None:
        sql = text(
            """
            SELECT
                id,
                dataset_code,
                name,
                version,
                domain,
                description,
                generator_provider,
                generator_model,
                reviewer_provider,
                reviewer_model,
                status,
                document_count,
                case_count,
                generation_config_json,
                content_sha256,
                frozen_at,
                created_at,
                updated_at
            FROM rag_eval_dataset
            WHERE dataset_code = :dataset_code
              AND version = :version
            LIMIT 1
            """
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                sql,
                {
                    "dataset_code": dataset_code,
                    "version": version,
                },
            ).mappings().first()
        if row is None:
            return None
        result = dict(row)
        config = result.get("generation_config_json")
        if isinstance(config, str):
            result["generation_config_json"] = json.loads(config)
        return result

    def save_source_document(
        self,
        dataset_id: int,
        document: SourceDocumentSpec,
    ) -> int:
        insert_sql = text(
            """
            INSERT INTO rag_eval_source_document (
                dataset_id,
                source_doc_code,
                title,
                doc_type,
                topic,
                version,
                effective_from,
                effective_to,
                is_current,
                relative_file_path,
                source_content_sha256,
                generation_spec_json,
                review_status,
                review_score,
                review_reason,
                mapped_doc_id
            ) VALUES (
                :dataset_id,
                :source_doc_code,
                :title,
                :doc_type,
                :topic,
                :version,
                :effective_from,
                :effective_to,
                :is_current,
                :relative_file_path,
                :source_content_sha256,
                CAST(:generation_spec_json AS JSON),
                :review_status,
                :review_score,
                :review_reason,
                :mapped_doc_id
            )
            """
        )
        count_sql = text(
            """
            UPDATE rag_eval_dataset
            SET document_count = (
                SELECT COUNT(*)
                FROM rag_eval_source_document
                WHERE dataset_id = :dataset_id
            )
            WHERE id = :dataset_id
            """
        )
        params = {
            "dataset_id": dataset_id,
            "source_doc_code": document.source_doc_code,
            "title": document.title,
            "doc_type": document.doc_type.value,
            "topic": document.topic,
            "version": document.version,
            "effective_from": document.effective_from,
            "effective_to": document.effective_to,
            "is_current": 1 if document.is_current else 0,
            "relative_file_path": document.relative_file_path,
            "source_content_sha256": document.source_content_sha256.lower(),
            "generation_spec_json": _json_dumps(document.generation_spec),
            "review_status": document.review_status.value,
            "review_score": document.review_score,
            "review_reason": document.review_reason,
            "mapped_doc_id": document.mapped_doc_id,
        }

        with self.engine.begin() as connection:
            result = connection.execute(insert_sql, params)
            connection.execute(count_sql, {"dataset_id": dataset_id})
            return int(result.lastrowid)

    def upsert_source_document(
        self,
        dataset_id: int,
        document: SourceDocumentSpec,
    ) -> int:
        sql = text(
            """
            INSERT INTO rag_eval_source_document (
                dataset_id,
                source_doc_code,
                title,
                doc_type,
                topic,
                version,
                effective_from,
                effective_to,
                is_current,
                relative_file_path,
                source_content_sha256,
                generation_spec_json,
                review_status,
                review_score,
                review_reason,
                mapped_doc_id
            ) VALUES (
                :dataset_id,
                :source_doc_code,
                :title,
                :doc_type,
                :topic,
                :version,
                :effective_from,
                :effective_to,
                :is_current,
                :relative_file_path,
                :source_content_sha256,
                CAST(:generation_spec_json AS JSON),
                :review_status,
                :review_score,
                :review_reason,
                :mapped_doc_id
            )
            ON DUPLICATE KEY UPDATE
                id = LAST_INSERT_ID(id),
                title = VALUES(title),
                doc_type = VALUES(doc_type),
                topic = VALUES(topic),
                version = VALUES(version),
                effective_from = VALUES(effective_from),
                effective_to = VALUES(effective_to),
                is_current = VALUES(is_current),
                relative_file_path = VALUES(relative_file_path),
                source_content_sha256 = VALUES(source_content_sha256),
                generation_spec_json = VALUES(generation_spec_json),
                review_status = VALUES(review_status),
                review_score = VALUES(review_score),
                review_reason = VALUES(review_reason),
                mapped_doc_id = COALESCE(
                    VALUES(mapped_doc_id),
                    mapped_doc_id
                )
            """
        )
        count_sql = text(
            """
            UPDATE rag_eval_dataset
            SET document_count = (
                SELECT COUNT(*)
                FROM rag_eval_source_document
                WHERE dataset_id = :dataset_id
            )
            WHERE id = :dataset_id
            """
        )
        params = {
            "dataset_id": dataset_id,
            "source_doc_code": document.source_doc_code,
            "title": document.title,
            "doc_type": document.doc_type.value,
            "topic": document.topic,
            "version": document.version,
            "effective_from": document.effective_from,
            "effective_to": document.effective_to,
            "is_current": 1 if document.is_current else 0,
            "relative_file_path": document.relative_file_path,
            "source_content_sha256": document.source_content_sha256.lower(),
            "generation_spec_json": _json_dumps(document.generation_spec),
            "review_status": document.review_status.value,
            "review_score": document.review_score,
            "review_reason": document.review_reason,
            "mapped_doc_id": document.mapped_doc_id,
        }

        with self.engine.begin() as connection:
            result = connection.execute(sql, params)
            connection.execute(count_sql, {"dataset_id": dataset_id})
            return int(result.lastrowid)

    def map_source_document(
        self,
        dataset_id: int,
        source_doc_code: str,
        mapped_doc_id: int,
    ) -> None:
        sql = text(
            """
            UPDATE rag_eval_source_document
            SET mapped_doc_id = :mapped_doc_id
            WHERE dataset_id = :dataset_id
              AND source_doc_code = :source_doc_code
            """
        )

        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "dataset_id": dataset_id,
                    "source_doc_code": source_doc_code,
                    "mapped_doc_id": mapped_doc_id,
                },
            )
            exists = connection.execute(
                text(
                    """
                    SELECT 1
                    FROM rag_eval_source_document
                    WHERE dataset_id = :dataset_id
                      AND source_doc_code = :source_doc_code
                    LIMIT 1
                    """
                ),
                {
                    "dataset_id": dataset_id,
                    "source_doc_code": source_doc_code,
                },
            ).first()

        if result.rowcount not in (0, 1) or exists is None:
            raise ValueError(
                f"未找到唯一源文档：dataset_id={dataset_id}, "
                f"source_doc_code={source_doc_code}"
            )

    def list_source_documents(
        self,
        dataset_id: int,
    ) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                id,
                dataset_id,
                source_doc_code,
                title,
                doc_type,
                topic,
                version,
                effective_from,
                effective_to,
                is_current,
                relative_file_path,
                source_content_sha256,
                generation_spec_json,
                review_status,
                review_score,
                review_reason,
                mapped_doc_id,
                created_at,
                updated_at
            FROM rag_eval_source_document
            WHERE dataset_id = :dataset_id
            ORDER BY source_doc_code ASC
            """
        )
        with self.engine.begin() as connection:
            rows = connection.execute(
                sql,
                {"dataset_id": dataset_id},
            ).mappings().all()
        result = []
        for row in rows:
            item = dict(row)
            item["generation_spec_json"] = _json_loads(
                item.get("generation_spec_json"),
                {},
            )
            result.append(item)
        return result

    def update_source_document_review(
        self,
        source_document_id: int,
        review_status: ReviewStatus,
        review_score: float | None,
        review_reason: str | None,
    ) -> None:
        sql = text(
            """
            UPDATE rag_eval_source_document
            SET review_status = :review_status,
                review_score = :review_score,
                review_reason = :review_reason
            WHERE id = :source_document_id
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "source_document_id": source_document_id,
                    "review_status": review_status.value,
                    "review_score": review_score,
                    "review_reason": review_reason,
                },
            )

        if result.rowcount not in (0, 1):
            raise ValueError(
                f"源文档审核状态更新异常：source_document_id={source_document_id}"
            )

    def save_eval_case(
        self,
        dataset_id: int,
        case: GeneratedEvalCase | ReviewedEvalCase,
    ) -> int:
        insert_sql = text(
            """
            INSERT INTO rag_eval_case (
                dataset_id,
                case_code,
                question,
                normalized_question,
                reference_answer,
                case_type,
                target_doc_types_json,
                expected_action,
                difficulty,
                dataset_split,
                business_domain,
                required_fact_count,
                generation_metadata_json,
                review_status,
                review_score,
                review_reason,
                status
            ) VALUES (
                :dataset_id,
                :case_code,
                :question,
                :normalized_question,
                :reference_answer,
                :case_type,
                CAST(:target_doc_types_json AS JSON),
                :expected_action,
                :difficulty,
                :dataset_split,
                :business_domain,
                :required_fact_count,
                CAST(:generation_metadata_json AS JSON),
                :review_status,
                :review_score,
                :review_reason,
                :status
            )
            """
        )
        count_sql = text(
            """
            UPDATE rag_eval_dataset
            SET case_count = (
                SELECT COUNT(*)
                FROM rag_eval_case
                WHERE dataset_id = :dataset_id
            )
            WHERE id = :dataset_id
            """
        )
        review_status = getattr(case, "review_status", ReviewStatus.PENDING)
        review_score = getattr(case, "review_score", None)
        review_reason = getattr(case, "review_reason", None)
        case_status = getattr(case, "status", EvalCaseStatus.ACTIVE)
        params = {
            "dataset_id": dataset_id,
            "case_code": case.case_code,
            "question": case.question,
            "normalized_question": case.normalized_question,
            "reference_answer": case.reference_answer,
            "case_type": case.case_type.value,
            "target_doc_types_json": _json_dumps(
                [item.value for item in case.target_doc_types]
            ),
            "expected_action": case.expected_action.value,
            "difficulty": case.difficulty.value,
            "dataset_split": case.dataset_split.value,
            "business_domain": case.business_domain,
            "required_fact_count": case.required_fact_count,
            "generation_metadata_json": _json_dumps(case.generation_metadata),
            "review_status": review_status.value,
            "review_score": review_score,
            "review_reason": review_reason,
            "status": case_status.value,
        }

        with self.engine.begin() as connection:
            result = connection.execute(insert_sql, params)
            connection.execute(count_sql, {"dataset_id": dataset_id})
            return int(result.lastrowid)

    def upsert_eval_case(
        self,
        dataset_id: int,
        case: GeneratedEvalCase | ReviewedEvalCase,
    ) -> int:
        sql = text(
            """
            INSERT INTO rag_eval_case (
                dataset_id,
                case_code,
                question,
                normalized_question,
                reference_answer,
                case_type,
                target_doc_types_json,
                expected_action,
                difficulty,
                dataset_split,
                business_domain,
                required_fact_count,
                generation_metadata_json,
                review_status,
                review_score,
                review_reason,
                status
            ) VALUES (
                :dataset_id,
                :case_code,
                :question,
                :normalized_question,
                :reference_answer,
                :case_type,
                CAST(:target_doc_types_json AS JSON),
                :expected_action,
                :difficulty,
                :dataset_split,
                :business_domain,
                :required_fact_count,
                CAST(:generation_metadata_json AS JSON),
                :review_status,
                :review_score,
                :review_reason,
                :status
            )
            ON DUPLICATE KEY UPDATE
                id = LAST_INSERT_ID(id),
                question = VALUES(question),
                normalized_question = VALUES(normalized_question),
                reference_answer = VALUES(reference_answer),
                case_type = VALUES(case_type),
                target_doc_types_json = VALUES(target_doc_types_json),
                expected_action = VALUES(expected_action),
                difficulty = VALUES(difficulty),
                dataset_split = VALUES(dataset_split),
                business_domain = VALUES(business_domain),
                required_fact_count = VALUES(required_fact_count),
                generation_metadata_json = VALUES(generation_metadata_json),
                review_status = VALUES(review_status),
                review_score = VALUES(review_score),
                review_reason = VALUES(review_reason),
                status = VALUES(status)
            """
        )
        review_status = getattr(case, "review_status", ReviewStatus.PENDING)
        review_score = getattr(case, "review_score", None)
        review_reason = getattr(case, "review_reason", None)
        case_status = getattr(case, "status", EvalCaseStatus.ACTIVE)
        params = {
            "dataset_id": dataset_id,
            "case_code": case.case_code,
            "question": case.question,
            "normalized_question": case.normalized_question,
            "reference_answer": case.reference_answer,
            "case_type": case.case_type.value,
            "target_doc_types_json": _json_dumps(
                [item.value for item in case.target_doc_types]
            ),
            "expected_action": case.expected_action.value,
            "difficulty": case.difficulty.value,
            "dataset_split": case.dataset_split.value,
            "business_domain": case.business_domain,
            "required_fact_count": case.required_fact_count,
            "generation_metadata_json": _json_dumps(case.generation_metadata),
            "review_status": review_status.value,
            "review_score": review_score,
            "review_reason": review_reason,
            "status": case_status.value,
        }
        count_sql = text(
            """
            UPDATE rag_eval_dataset
            SET case_count = (
                SELECT COUNT(*)
                FROM rag_eval_case
                WHERE dataset_id = :dataset_id
            )
            WHERE id = :dataset_id
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(sql, params)
            connection.execute(count_sql, {"dataset_id": dataset_id})
            return int(result.lastrowid)

    def update_eval_case_review(
        self,
        case_id: int,
        review_status: ReviewStatus,
        review_score: float | None,
        review_reason: str | None,
        status: EvalCaseStatus = EvalCaseStatus.ACTIVE,
    ) -> None:
        sql = text(
            """
            UPDATE rag_eval_case
            SET review_status = :review_status,
                review_score = :review_score,
                review_reason = :review_reason,
                status = :status
            WHERE id = :case_id
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "case_id": case_id,
                    "review_status": review_status.value,
                    "review_score": review_score,
                    "review_reason": review_reason,
                    "status": status.value,
                },
            )

        if result.rowcount not in (0, 1):
            raise ValueError(f"评测题审核状态更新异常：case_id={case_id}")

    def save_case_evidence(
        self,
        case_id: int,
        source_document_id: int,
        evidence: EvidenceSpec,
    ) -> int:
        normalized_quote = " ".join(evidence.evidence_quote.split())
        quote_sha256 = hashlib.sha256(
            normalized_quote.encode("utf-8")
        ).hexdigest()
        sql = text(
            """
            INSERT INTO rag_eval_case_relevance (
                case_id,
                source_document_id,
                mapped_doc_id,
                mapped_chunk_id,
                relevance_grade,
                evidence_quote,
                evidence_quote_sha256,
                fact_key,
                mapping_status,
                mapping_reason
            ) VALUES (
                :case_id,
                :source_document_id,
                :mapped_doc_id,
                :mapped_chunk_id,
                :relevance_grade,
                :evidence_quote,
                :evidence_quote_sha256,
                :fact_key,
                :mapping_status,
                :mapping_reason
            )
            """
        )
        params = {
            "case_id": case_id,
            "source_document_id": source_document_id,
            "mapped_doc_id": evidence.mapped_doc_id,
            "mapped_chunk_id": evidence.mapped_chunk_id,
            "relevance_grade": evidence.relevance_grade,
            "evidence_quote": evidence.evidence_quote,
            "evidence_quote_sha256": quote_sha256,
            "fact_key": evidence.fact_key,
            "mapping_status": evidence.mapping_status.value,
            "mapping_reason": evidence.mapping_reason,
        }

        with self.engine.begin() as connection:
            result = connection.execute(sql, params)
            return int(result.lastrowid)

    def delete_case_evidence(self, case_id: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    DELETE FROM rag_eval_case_relevance
                    WHERE case_id = :case_id
                    """
                ),
                {"case_id": case_id},
            )

    def map_case_evidence(
        self,
        evidence_id: int,
        mapped_doc_id: int,
        mapped_chunk_id: int,
    ) -> None:
        self.update_case_evidence_mapping(
            evidence_id=evidence_id,
            mapping_status=MappingStatus.MAPPED,
            mapped_doc_id=mapped_doc_id,
            mapped_chunk_id=mapped_chunk_id,
            mapping_reason=None,
        )

    def update_case_evidence_mapping(
        self,
        evidence_id: int,
        mapping_status: MappingStatus,
        mapped_doc_id: int | None = None,
        mapped_chunk_id: int | None = None,
        mapping_reason: str | None = None,
    ) -> None:
        if mapping_status == MappingStatus.MAPPED:
            if mapped_doc_id is None or mapped_chunk_id is None:
                raise ValueError("MAPPED 状态必须提供 mapped_doc_id 和 mapped_chunk_id")

        sql = text(
            """
            UPDATE rag_eval_case_relevance
            SET mapped_doc_id = :mapped_doc_id,
                mapped_chunk_id = :mapped_chunk_id,
                mapping_status = :mapping_status,
                mapping_reason = :mapping_reason
            WHERE id = :evidence_id
            """
        )

        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "evidence_id": evidence_id,
                    "mapped_doc_id": mapped_doc_id,
                    "mapped_chunk_id": mapped_chunk_id,
                    "mapping_status": mapping_status.value,
                    "mapping_reason": mapping_reason,
                },
            )
            exists = connection.execute(
                text(
                    """
                    SELECT 1
                    FROM rag_eval_case_relevance
                    WHERE id = :evidence_id
                    LIMIT 1
                    """
                ),
                {"evidence_id": evidence_id},
            ).first()

        if result.rowcount not in (0, 1) or exists is None:
            raise ValueError(f"未找到标准证据：evidence_id={evidence_id}")

    def list_case_evidence(self, case_id: int) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                relevance.id,
                relevance.case_id,
                relevance.source_document_id,
                source.source_doc_code,
                relevance.mapped_doc_id,
                relevance.mapped_chunk_id,
                relevance.relevance_grade,
                relevance.evidence_quote,
                relevance.evidence_quote_sha256,
                relevance.fact_key,
                relevance.mapping_status,
                relevance.mapping_reason
            FROM rag_eval_case_relevance AS relevance
            JOIN rag_eval_source_document AS source
              ON source.id = relevance.source_document_id
            WHERE relevance.case_id = :case_id
            ORDER BY relevance.relevance_grade DESC, relevance.id
            """
        )

        with self.engine.begin() as connection:
            rows = connection.execute(
                sql,
                {"case_id": case_id},
            ).mappings().all()

        return [dict(row) for row in rows]

    def list_reviewed_cases(
        self,
        dataset_id: int,
        split: DatasetSplit,
    ) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                id,
                dataset_id,
                case_code,
                question,
                normalized_question,
                reference_answer,
                case_type,
                target_doc_types_json,
                expected_action,
                difficulty,
                dataset_split,
                business_domain,
                required_fact_count,
                generation_metadata_json,
                review_status,
                review_score,
                review_reason,
                status
            FROM rag_eval_case
            WHERE dataset_id = :dataset_id
              AND dataset_split = :dataset_split
              AND review_status = 'PASSED'
              AND status = 'ACTIVE'
            ORDER BY id
            """
        )

        with self.engine.begin() as connection:
            rows = connection.execute(
                sql,
                {
                    "dataset_id": dataset_id,
                    "dataset_split": split.value,
                },
            ).mappings().all()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["target_doc_types"] = _json_loads(
                item.pop("target_doc_types_json"),
                [],
            )
            item["generation_metadata"] = _json_loads(
                item.pop("generation_metadata_json"),
                {},
            )
            results.append(item)

        return results

    def create_run(self, config: EvalRunConfig) -> int:
        sql = text(
            """
            INSERT INTO rag_eval_run (
                run_code,
                dataset_id,
                experiment_version,
                experiment_name,
                git_commit_sha,
                retrieval_mode,
                embedding_model,
                rerank_model,
                answer_model,
                judge_model,
                config_json,
                status,
                total_cases
            ) VALUES (
                :run_code,
                :dataset_id,
                :experiment_version,
                :experiment_name,
                :git_commit_sha,
                :retrieval_mode,
                :embedding_model,
                :rerank_model,
                :answer_model,
                :judge_model,
                CAST(:config_json AS JSON),
                'PENDING',
                :total_cases
            )
            """
        )
        params = {
            "run_code": config.run_code,
            "dataset_id": config.dataset_id,
            "experiment_version": config.experiment_version,
            "experiment_name": config.experiment_name,
            "git_commit_sha": config.git_commit_sha,
            "retrieval_mode": config.retrieval_mode,
            "embedding_model": config.embedding_model,
            "rerank_model": config.rerank_model,
            "answer_model": config.answer_model,
            "judge_model": config.judge_model,
            "config_json": _json_dumps(config.config),
            "total_cases": config.total_cases,
        }

        with self.engine.begin() as connection:
            result = connection.execute(sql, params)
            return int(result.lastrowid)

    def find_run_by_code(
        self,
        run_code: str,
    ) -> dict[str, Any] | None:
        sql = text(
            """
            SELECT
                id,
                run_code,
                dataset_id,
                experiment_version,
                experiment_name,
                git_commit_sha,
                retrieval_mode,
                embedding_model,
                rerank_model,
                answer_model,
                judge_model,
                config_json,
                status,
                total_cases,
                completed_cases,
                failed_cases,
                summary_metrics_json,
                started_at,
                finished_at,
                error_message
            FROM rag_eval_run
            WHERE run_code = :run_code
            LIMIT 1
            """
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                sql,
                {"run_code": run_code},
            ).mappings().first()
        if row is None:
            return None
        result = dict(row)
        result["config"] = _json_loads(
            result.pop("config_json"),
            {},
        )
        result["summary_metrics"] = _json_loads(
            result.pop("summary_metrics_json"),
            {},
        )
        return result

    def update_dataset_status(
        self,
        dataset_id: int,
        status: DatasetStatus,
    ) -> None:
        sql = text(
            """
            UPDATE rag_eval_dataset
            SET status = :status
            WHERE id = :dataset_id
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "dataset_id": dataset_id,
                    "status": status.value,
                },
            )

        if result.rowcount not in (0, 1):
            raise ValueError(f"数据集状态更新异常：dataset_id={dataset_id}")

    def freeze_dataset(
        self,
        dataset_id: int,
        content_sha256: str,
    ) -> None:
        normalized_sha256 = content_sha256.lower()
        if (
            len(normalized_sha256) != 64
            or any(char not in "0123456789abcdef" for char in normalized_sha256)
        ):
            raise ValueError("content_sha256 必须是 64 位十六进制字符串")

        sql = text(
            """
            UPDATE rag_eval_dataset
            SET status = 'FROZEN',
                content_sha256 = :content_sha256,
                frozen_at = CURRENT_TIMESTAMP
            WHERE id = :dataset_id
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "dataset_id": dataset_id,
                    "content_sha256": normalized_sha256,
                },
            )

        if result.rowcount not in (0, 1):
            raise ValueError(f"数据集冻结状态更新异常：dataset_id={dataset_id}")

    def start_run(self, run_id: int) -> None:
        sql = text(
            """
            UPDATE rag_eval_run
            SET status = 'RUNNING',
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                error_message = NULL
            WHERE id = :run_id
            """
        )
        with self.engine.begin() as connection:
            connection.execute(sql, {"run_id": run_id})

    def start_case_result(
        self,
        run_id: int,
        case_id: int,
        trace_id: str | None = None,
    ) -> int:
        sql = text(
            """
            INSERT INTO rag_eval_case_result (
                run_id,
                case_id,
                trace_id,
                status,
                started_at
            ) VALUES (
                :run_id,
                :case_id,
                :trace_id,
                'PENDING',
                CURRENT_TIMESTAMP
            )
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                sql,
                {
                    "run_id": run_id,
                    "case_id": case_id,
                    "trace_id": trace_id,
                },
            )
            return int(result.lastrowid)

    def prepare_case_result(
        self,
        run_id: int,
        case_id: int,
    ) -> tuple[int, bool]:
        select_sql = text(
            """
            SELECT
                result.id,
                result.status,
                result.actual_action,
                EXISTS (
                    SELECT 1
                    FROM rag_eval_judge_result AS judge
                    WHERE judge.case_result_id = result.id
                ) AS has_judge
            FROM rag_eval_case_result AS result
            WHERE result.run_id = :run_id
              AND result.case_id = :case_id
            LIMIT 1
            FOR UPDATE
            """
        )
        insert_sql = text(
            """
            INSERT INTO rag_eval_case_result (
                run_id,
                case_id,
                status,
                started_at
            ) VALUES (
                :run_id,
                :case_id,
                'PENDING',
                CURRENT_TIMESTAMP
            )
            """
        )
        reset_sql = text(
            """
            UPDATE rag_eval_case_result
            SET trace_id = NULL,
                actual_action = NULL,
                generated_answer = NULL,
                retrieved_chunk_ids_json = NULL,
                cited_chunk_ids_json = NULL,
                recall_at_1 = NULL,
                recall_at_3 = NULL,
                recall_at_5 = NULL,
                recall_at_10 = NULL,
                reciprocal_rank = NULL,
                ndcg_at_5 = NULL,
                ndcg_at_10 = NULL,
                fact_coverage = NULL,
                citation_precision = NULL,
                citation_recall = NULL,
                action_correct = NULL,
                retrieval_rounds = 1,
                input_tokens = NULL,
                output_tokens = NULL,
                estimated_cost = NULL,
                latency_ms = NULL,
                status = 'PENDING',
                error_message = NULL,
                started_at = CURRENT_TIMESTAMP,
                finished_at = NULL
            WHERE id = :case_result_id
            """
        )
        params = {"run_id": run_id, "case_id": case_id}
        with self.engine.begin() as connection:
            row = connection.execute(
                select_sql,
                params,
            ).mappings().first()
            if row is None:
                result = connection.execute(insert_sql, params)
                return int(result.lastrowid), True

            case_result_id = int(row["id"])
            is_complete_success = (
                row["status"] == EvalCaseResultStatus.SUCCESS.value
                and row["actual_action"] != ActualAction.ERROR.value
                and bool(row["has_judge"])
            )
            if is_complete_success:
                return case_result_id, False

            connection.execute(
                text(
                    """
                    DELETE FROM rag_eval_judge_result
                    WHERE case_result_id = :case_result_id
                    """
                ),
                {"case_result_id": case_result_id},
            )
            connection.execute(
                text(
                    """
                    DELETE FROM rag_eval_retrieval_hit
                    WHERE case_result_id = :case_result_id
                    """
                ),
                {"case_result_id": case_result_id},
            )
            connection.execute(
                reset_sql,
                {"case_result_id": case_result_id},
            )
            return case_result_id, True

    def update_case_result_trace(
        self,
        case_result_id: int,
        trace_id: str,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE rag_eval_case_result
                    SET trace_id = :trace_id
                    WHERE id = :case_result_id
                    """
                ),
                {
                    "case_result_id": case_result_id,
                    "trace_id": trace_id,
                },
            )

    def save_retrieval_hits(
        self,
        case_result_id: int,
        hits: list[dict[str, Any]],
    ) -> None:
        sql = text(
            """
            INSERT INTO rag_eval_retrieval_hit (
                case_result_id,
                retrieval_round,
                query_variant,
                query_text,
                channel,
                chunk_id,
                rank_no,
                raw_score,
                fused_score,
                rerank_score,
                is_gold,
                metadata_json
            ) VALUES (
                :case_result_id,
                :retrieval_round,
                :query_variant,
                :query_text,
                :channel,
                :chunk_id,
                :rank_no,
                :raw_score,
                :fused_score,
                :rerank_score,
                :is_gold,
                CAST(:metadata_json AS JSON)
            )
            """
        )

        with self.engine.begin() as connection:
            for hit in hits:
                connection.execute(
                    sql,
                    {
                        "case_result_id": case_result_id,
                        "retrieval_round": hit.get("retrieval_round", 1),
                        "query_variant": hit.get("query_variant", "ORIGINAL"),
                        "query_text": hit["query_text"],
                        "channel": hit["channel"],
                        "chunk_id": hit["chunk_id"],
                        "rank_no": hit["rank_no"],
                        "raw_score": hit.get("raw_score"),
                        "fused_score": hit.get("fused_score"),
                        "rerank_score": hit.get("rerank_score"),
                        "is_gold": 1 if hit.get("is_gold") else 0,
                        "metadata_json": _json_dumps(hit.get("metadata") or {}),
                    },
                )

    def finish_case_result(
        self,
        case_result_id: int,
        actual_action: ActualAction,
        generated_answer: str | None,
        retrieved_chunk_ids: list[int],
        cited_chunk_ids: list[int],
        metrics: RetrievalMetricResult,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost: float | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
    ) -> None:
        status = (
            "FAILED"
            if error_message is not None
            else "SUCCESS"
        )
        sql = text(
            """
            UPDATE rag_eval_case_result
            SET actual_action = :actual_action,
                generated_answer = :generated_answer,
                retrieved_chunk_ids_json = CAST(:retrieved_chunk_ids_json AS JSON),
                cited_chunk_ids_json = CAST(:cited_chunk_ids_json AS JSON),
                recall_at_1 = :recall_at_1,
                recall_at_3 = :recall_at_3,
                recall_at_5 = :recall_at_5,
                recall_at_10 = :recall_at_10,
                reciprocal_rank = :reciprocal_rank,
                ndcg_at_5 = :ndcg_at_5,
                ndcg_at_10 = :ndcg_at_10,
                fact_coverage = :fact_coverage,
                citation_precision = :citation_precision,
                citation_recall = :citation_recall,
                action_correct = :action_correct,
                retrieval_rounds = :retrieval_rounds,
                input_tokens = :input_tokens,
                output_tokens = :output_tokens,
                estimated_cost = :estimated_cost,
                latency_ms = :latency_ms,
                status = :status,
                error_message = :error_message,
                finished_at = CURRENT_TIMESTAMP
            WHERE id = :case_result_id
            """
        )
        params = {
            "case_result_id": case_result_id,
            "actual_action": actual_action.value,
            "generated_answer": generated_answer,
            "retrieved_chunk_ids_json": _json_dumps(retrieved_chunk_ids),
            "cited_chunk_ids_json": _json_dumps(cited_chunk_ids),
            "recall_at_1": metrics.recall_at_1,
            "recall_at_3": metrics.recall_at_3,
            "recall_at_5": metrics.recall_at_5,
            "recall_at_10": metrics.recall_at_10,
            "reciprocal_rank": metrics.reciprocal_rank,
            "ndcg_at_5": metrics.ndcg_at_5,
            "ndcg_at_10": metrics.ndcg_at_10,
            "fact_coverage": metrics.fact_coverage,
            "citation_precision": metrics.citation_precision,
            "citation_recall": metrics.citation_recall,
            "action_correct": (
                None
                if metrics.action_correct is None
                else 1 if metrics.action_correct else 0
            ),
            "retrieval_rounds": metrics.retrieval_rounds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
            "latency_ms": latency_ms,
            "status": status,
            "error_message": error_message,
        }
        with self.engine.begin() as connection:
            connection.execute(sql, params)

    def save_judge_result(
        self,
        case_result_id: int,
        score: JudgeScore,
    ) -> int:
        sql = text(
            """
            INSERT INTO rag_eval_judge_result (
                case_result_id,
                judge_provider,
                judge_model,
                judge_prompt_version,
                faithfulness_score,
                answer_relevance_score,
                completeness_score,
                citation_entailment_score,
                conflict_handling_score,
                refusal_correct,
                clarification_correct,
                passed,
                reason_json,
                raw_response_json,
                latency_ms
            ) VALUES (
                :case_result_id,
                :judge_provider,
                :judge_model,
                :judge_prompt_version,
                :faithfulness_score,
                :answer_relevance_score,
                :completeness_score,
                :citation_entailment_score,
                :conflict_handling_score,
                :refusal_correct,
                :clarification_correct,
                :passed,
                CAST(:reason_json AS JSON),
                CAST(:raw_response_json AS JSON),
                :latency_ms
            )
            ON DUPLICATE KEY UPDATE
                id = LAST_INSERT_ID(id),
                faithfulness_score = VALUES(faithfulness_score),
                answer_relevance_score = VALUES(answer_relevance_score),
                completeness_score = VALUES(completeness_score),
                citation_entailment_score = VALUES(citation_entailment_score),
                conflict_handling_score = VALUES(conflict_handling_score),
                refusal_correct = VALUES(refusal_correct),
                clarification_correct = VALUES(clarification_correct),
                passed = VALUES(passed),
                reason_json = VALUES(reason_json),
                raw_response_json = VALUES(raw_response_json),
                latency_ms = VALUES(latency_ms)
            """
        )
        params = {
            "case_result_id": case_result_id,
            "judge_provider": score.judge_provider,
            "judge_model": score.judge_model,
            "judge_prompt_version": score.judge_prompt_version,
            "faithfulness_score": score.faithfulness_score,
            "answer_relevance_score": score.answer_relevance_score,
            "completeness_score": score.completeness_score,
            "citation_entailment_score": score.citation_entailment_score,
            "conflict_handling_score": score.conflict_handling_score,
            "refusal_correct": (
                None
                if score.refusal_correct is None
                else 1 if score.refusal_correct else 0
            ),
            "clarification_correct": (
                None
                if score.clarification_correct is None
                else 1 if score.clarification_correct else 0
            ),
            "passed": 1 if score.passed else 0,
            "reason_json": _json_dumps(score.reason),
            "raw_response_json": _json_dumps(score.raw_response),
            "latency_ms": score.latency_ms,
        }
        with self.engine.begin() as connection:
            result = connection.execute(sql, params)
            return int(result.lastrowid)

    def list_run_case_results(
        self,
        run_id: int,
    ) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                result.id,
                result.run_id,
                result.case_id,
                eval_case.case_code,
                eval_case.question,
                eval_case.reference_answer,
                eval_case.case_type,
                eval_case.expected_action,
                eval_case.difficulty,
                eval_case.dataset_split,
                eval_case.business_domain,
                eval_case.required_fact_count,
                result.trace_id,
                result.actual_action,
                result.generated_answer,
                result.retrieved_chunk_ids_json,
                result.cited_chunk_ids_json,
                result.recall_at_1,
                result.recall_at_3,
                result.recall_at_5,
                result.recall_at_10,
                result.reciprocal_rank,
                result.ndcg_at_5,
                result.ndcg_at_10,
                result.fact_coverage,
                result.citation_precision,
                result.citation_recall,
                result.action_correct,
                result.retrieval_rounds,
                result.input_tokens,
                result.output_tokens,
                result.estimated_cost,
                result.latency_ms,
                result.status,
                result.error_message,
                judge.passed AS judge_passed,
                judge.faithfulness_score,
                judge.answer_relevance_score,
                judge.completeness_score,
                judge.citation_entailment_score,
                judge.conflict_handling_score,
                judge.refusal_correct,
                judge.clarification_correct,
                judge.reason_json AS judge_reason_json
            FROM rag_eval_case_result AS result
            JOIN rag_eval_case AS eval_case
              ON eval_case.id = result.case_id
            LEFT JOIN rag_eval_judge_result AS judge
              ON judge.id = (
                  SELECT MAX(candidate.id)
                  FROM rag_eval_judge_result AS candidate
                  WHERE candidate.case_result_id = result.id
              )
            WHERE result.run_id = :run_id
            ORDER BY result.id
            """
        )
        with self.engine.begin() as connection:
            rows = connection.execute(
                sql,
                {"run_id": run_id},
            ).mappings().all()
        results = []
        for row in rows:
            item = dict(row)
            item["retrieved_chunk_ids"] = _json_loads(
                item.pop("retrieved_chunk_ids_json"),
                [],
            )
            item["cited_chunk_ids"] = _json_loads(
                item.pop("cited_chunk_ids_json"),
                [],
            )
            item["judge_reason"] = _json_loads(
                item.pop("judge_reason_json"),
                {},
            )
            results.append(item)
        return results

    def list_run_retrieval_hits(
        self,
        run_id: int,
    ) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                hit.id,
                hit.case_result_id,
                result.case_id,
                hit.retrieval_round,
                hit.query_variant,
                hit.query_text,
                hit.channel,
                hit.chunk_id,
                hit.rank_no,
                hit.raw_score,
                hit.fused_score,
                hit.rerank_score,
                hit.is_gold,
                hit.metadata_json
            FROM rag_eval_retrieval_hit AS hit
            JOIN rag_eval_case_result AS result
              ON result.id = hit.case_result_id
            WHERE result.run_id = :run_id
            ORDER BY
                hit.case_result_id,
                hit.retrieval_round,
                hit.channel,
                hit.rank_no,
                hit.id
            """
        )
        with self.engine.begin() as connection:
            rows = connection.execute(
                sql,
                {"run_id": run_id},
            ).mappings().all()
        results = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _json_loads(
                item.pop("metadata_json"),
                {},
            )
            results.append(item)
        return results

    def list_run_evidences(
        self,
        run_id: int,
    ) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                result.case_id,
                eval_case.case_code,
                relevance.id AS evidence_id,
                source.source_doc_code,
                source.title AS source_title,
                source.doc_type,
                relevance.mapped_doc_id,
                relevance.mapped_chunk_id,
                relevance.relevance_grade,
                relevance.evidence_quote,
                relevance.fact_key,
                relevance.mapping_status,
                relevance.mapping_reason
            FROM rag_eval_case_result AS result
            JOIN rag_eval_case AS eval_case
              ON eval_case.id = result.case_id
            JOIN rag_eval_case_relevance AS relevance
              ON relevance.case_id = result.case_id
            JOIN rag_eval_source_document AS source
              ON source.id = relevance.source_document_id
            WHERE result.run_id = :run_id
            ORDER BY
                result.case_id,
                relevance.relevance_grade DESC,
                relevance.id
            """
        )
        with self.engine.begin() as connection:
            rows = connection.execute(
                sql,
                {"run_id": run_id},
            ).mappings().all()
        return [dict(row) for row in rows]

    def get_run_domain_diagnostics(
        self,
        run_id: int,
    ) -> dict[str, Any]:
        case_domain_sql = text(
            """
            SELECT
                eval_case.business_domain,
                COUNT(*) AS case_count
            FROM rag_eval_case_result AS result
            JOIN rag_eval_case AS eval_case
              ON eval_case.id = result.case_id
            WHERE result.run_id = :run_id
            GROUP BY eval_case.business_domain
            ORDER BY case_count DESC, eval_case.business_domain
            """
        )
        chunk_domain_sql = text(
            """
            SELECT
                business_domain,
                COUNT(*) AS chunk_count
            FROM rag_chunk
            WHERE status = 'ACTIVE'
            GROUP BY business_domain
            ORDER BY chunk_count DESC, business_domain
            """
        )
        gold_domain_sql = text(
            """
            SELECT
                chunk.business_domain,
                COUNT(DISTINCT relevance.mapped_chunk_id) AS chunk_count
            FROM rag_eval_case_result AS result
            JOIN rag_eval_case_relevance AS relevance
              ON relevance.case_id = result.case_id
            LEFT JOIN rag_chunk AS chunk
              ON chunk.id = relevance.mapped_chunk_id
            WHERE result.run_id = :run_id
            GROUP BY chunk.business_domain
            ORDER BY chunk_count DESC, chunk.business_domain
            """
        )
        hit_count_sql = text(
            """
            SELECT COUNT(*)
            FROM rag_eval_retrieval_hit AS hit
            JOIN rag_eval_case_result AS result
              ON result.id = hit.case_result_id
            WHERE result.run_id = :run_id
            """
        )
        params = {"run_id": run_id}
        with self.engine.begin() as connection:
            case_domains = connection.execute(
                case_domain_sql,
                params,
            ).mappings().all()
            chunk_domains = connection.execute(
                chunk_domain_sql,
            ).mappings().all()
            gold_domains = connection.execute(
                gold_domain_sql,
                params,
            ).mappings().all()
            retrieval_hit_count = connection.execute(
                hit_count_sql,
                params,
            ).scalar_one()
        normalized_case_domains = [dict(row) for row in case_domains]
        return {
            "retrieval_hit_count": int(retrieval_hit_count),
            "case_domains": normalized_case_domains,
            "resolved_case_domains": list(
                resolve_business_domains(
                    [
                        item["business_domain"]
                        for item in normalized_case_domains
                        if item.get("business_domain")
                    ]
                )
            ),
            "active_chunk_domains": [dict(row) for row in chunk_domains],
            "gold_chunk_domains": [dict(row) for row in gold_domains],
        }

    def finish_run(
        self,
        run_id: int,
        status: EvalRunStatus,
        completed_cases: int,
        failed_cases: int,
        summary_metrics: dict[str, Any],
        error_message: str | None = None,
    ) -> None:
        sql = text(
            """
            UPDATE rag_eval_run
            SET status = :status,
                completed_cases = :completed_cases,
                failed_cases = :failed_cases,
                summary_metrics_json = CAST(:summary_metrics_json AS JSON),
                error_message = :error_message,
                finished_at = CURRENT_TIMESTAMP
            WHERE id = :run_id
            """
        )
        with self.engine.begin() as connection:
            connection.execute(
                sql,
                {
                    "run_id": run_id,
                    "status": status.value,
                    "completed_cases": completed_cases,
                    "failed_cases": failed_cases,
                    "summary_metrics_json": _json_dumps(summary_metrics),
                    "error_message": error_message,
                },
            )

    def delete_run(self, run_id: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM rag_eval_run WHERE id = :run_id"),
                {"run_id": run_id},
            )

    def delete_dataset(self, dataset_id: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM rag_eval_dataset WHERE id = :dataset_id"),
                {"dataset_id": dataset_id},
            )
