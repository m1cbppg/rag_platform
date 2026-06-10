import re
from pathlib import Path
from typing import Any
import hashlib

from pydantic import BaseModel, Field

from src.rag_platform.domain.document import DocumentStatus, DocumentType
from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    GeneratedSourceDocument,
)


class CorpusIngestResult(BaseModel):
    source_doc_code: str
    mapped_doc_id: int
    chunk_ids: list[int]
    embedding_task_summary: dict[str, int]
    keyword_task_summary: dict[str, int]
    fact_mappings: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationCorpusIngestService:
    def __init__(
        self,
        *,
        ingest_service,
        chunk_service,
        embedding_service,
        search_service,
        document_repository,
        dataset_repository,
        task_limit: int = 1000,
    ) -> None:
        self.ingest_service = ingest_service
        self.chunk_service = chunk_service
        self.embedding_service = embedding_service
        self.search_service = search_service
        self.document_repository = document_repository
        self.dataset_repository = dataset_repository
        self.task_limit = task_limit

    async def ingest_document(
        self,
        *,
        dataset_id: int,
        blueprint: DocumentBlueprint,
        document: GeneratedSourceDocument,
        rendered_path: Path,
    ) -> CorpusIngestResult:
        ingest_result = self.ingest_service.ingest_file_path(
            file_path=rendered_path,
            title=blueprint.title,
            doc_type=DocumentType(blueprint.doc_type.value),
            business_domain=blueprint.topic,
            version=blueprint.version,
            created_by="rag_eval_m3",
        )
        doc_id = int(ingest_result.doc_id)
        status = str(ingest_result.status)

        if status == DocumentStatus.CLEANED.value:
            chunk_result = self.chunk_service.build_chunks_for_document(doc_id)
            if chunk_result.status != DocumentStatus.CHUNKED.value:
                raise RuntimeError(
                    f"{blueprint.source_doc_code} Chunk构建失败："
                    f"status={chunk_result.status}"
                )
        elif status != DocumentStatus.CHUNKED.value:
            raise RuntimeError(
                f"{blueprint.source_doc_code} 文档状态不可继续：{status}"
            )

        chunks = self.document_repository.list_chunks_by_doc_id(doc_id)
        if not chunks:
            raise RuntimeError(f"{blueprint.source_doc_code} 未生成任何Chunk")

        fact_mappings = _validate_and_map_facts(
            blueprint=blueprint,
            document=document,
            chunks=chunks,
        )

        self.embedding_service.create_tasks(
            doc_id=doc_id,
            limit=self.task_limit,
        )
        embedding_run = await self.embedding_service.run_tasks(
            limit=self.task_limit,
            doc_id=doc_id,
        )
        if embedding_run.failed_count:
            raise RuntimeError(
                f"{blueprint.source_doc_code} Embedding失败"
                f"{embedding_run.failed_count}个任务"
            )
        embedding_summary = self.embedding_service.get_task_summary(doc_id)
        _require_all_success(
            pipeline_name="Embedding",
            source_doc_code=blueprint.source_doc_code,
            expected_count=len(chunks),
            summary=embedding_summary,
        )

        self.search_service.create_tasks(
            doc_id=doc_id,
            limit=self.task_limit,
        )
        search_run = self.search_service.run_tasks(
            limit=self.task_limit,
            doc_id=doc_id,
        )
        if search_run.failed_count:
            raise RuntimeError(
                f"{blueprint.source_doc_code} Elasticsearch失败"
                f"{search_run.failed_count}个任务"
            )
        keyword_summary = self.search_service.get_task_summary(doc_id)
        _require_all_success(
            pipeline_name="Elasticsearch",
            source_doc_code=blueprint.source_doc_code,
            expected_count=len(chunks),
            summary=keyword_summary,
        )

        self.dataset_repository.map_source_document(
            dataset_id=dataset_id,
            source_doc_code=blueprint.source_doc_code,
            mapped_doc_id=doc_id,
        )
        return CorpusIngestResult(
            source_doc_code=blueprint.source_doc_code,
            mapped_doc_id=doc_id,
            chunk_ids=[int(chunk["id"]) for chunk in chunks],
            embedding_task_summary=embedding_summary,
            keyword_task_summary=keyword_summary,
            fact_mappings=fact_mappings,
        )


def build_catalog_document(
    *,
    blueprint: DocumentBlueprint,
    document: GeneratedSourceDocument,
    mapped_doc_id: int,
    chunks: list[dict],
    source_content_sha256: str,
    rendered_path: Path | None = None,
) -> dict[str, Any]:
    fact_mappings = _validate_and_map_facts(
        blueprint=blueprint,
        document=document,
        chunks=chunks,
    )
    return {
        "source_doc_code": blueprint.source_doc_code,
        "mapped_doc_id": mapped_doc_id,
        "doc_type": blueprint.doc_type.value,
        "title": blueprint.title,
        "topic": blueprint.topic,
        "version": blueprint.version,
        "source_content_sha256": source_content_sha256,
        "rendered_file_sha256": (
            hashlib.sha256(rendered_path.read_bytes()).hexdigest()
            if rendered_path is not None
            else None
        ),
        "chunk_ids": [int(chunk["id"]) for chunk in chunks],
        "chunks": [
            {
                "chunk_id": int(chunk["id"]),
                "title": chunk.get("title"),
                "title_path": chunk.get("title_path"),
                "summary": chunk.get("summary"),
                "source_section": chunk.get("source_section"),
                "sort_order": chunk.get("sort_order"),
            }
            for chunk in chunks
        ],
        "facts": fact_mappings,
    }


def _validate_and_map_facts(
    *,
    blueprint: DocumentBlueprint,
    document: GeneratedSourceDocument,
    chunks: list[dict],
) -> list[dict[str, Any]]:
    normalized_chunks = {
        int(chunk["id"]): _normalize(str(chunk.get("content") or ""))
        for chunk in chunks
    }
    combined_text = "".join(normalized_chunks.values())
    missing_identifiers = [
        identifier
        for identifier in blueprint.required_identifiers
        if _normalize(identifier) not in combined_text
    ]
    if missing_identifiers:
        raise RuntimeError(
            f"{blueprint.source_doc_code} Chunk缺少必要标识符："
            f"{missing_identifiers}"
        )

    mappings: list[dict[str, Any]] = []
    for section in document.sections:
        for fact in section.facts:
            normalized_fact = _normalize(fact.fact_text)
            chunk_ids = [
                chunk_id
                for chunk_id, content in normalized_chunks.items()
                if normalized_fact in content
            ]
            if not chunk_ids:
                raise RuntimeError(
                    f"{blueprint.source_doc_code} Chunk缺少事实"
                    f" {fact.fact_key}: {fact.fact_text}"
                )
            mappings.append(
                {
                    "fact_key": fact.fact_key,
                    "fact_text": fact.fact_text,
                    "chunk_ids": chunk_ids,
                }
            )
    return mappings


def _require_all_success(
    *,
    pipeline_name: str,
    source_doc_code: str,
    expected_count: int,
    summary: dict[str, int],
) -> None:
    success_count = int(summary.get("SUCCESS", 0))
    non_success_count = sum(
        count
        for status, count in summary.items()
        if status != "SUCCESS"
    )
    if success_count != expected_count or non_success_count:
        raise RuntimeError(
            f"{source_doc_code} {pipeline_name}任务未全部成功："
            f"expected={expected_count}, summary={summary}"
        )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
