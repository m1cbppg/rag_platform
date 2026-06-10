import re
from dataclasses import dataclass
from typing import Any

from src.rag_platform.evaluation.models import (
    EvidenceSpec,
    MappingStatus,
    ReviewedEvalCase,
)


@dataclass(frozen=True)
class EvidenceMappingResult:
    status: MappingStatus
    chunk_id: int | None
    method: str
    reason: str


class EvidenceMapper:
    def map_quote(
        self,
        *,
        quote: str,
        chunks: list[dict[str, Any]],
    ) -> EvidenceMappingResult:
        exact_matches = self._find_matches(
            quote=quote,
            chunks=chunks,
            normalizer=_normalize_whitespace,
        )
        result = self._resolve_matches(exact_matches, "EXACT")
        if result is not None:
            return result

        normalized_matches = self._find_matches(
            quote=quote,
            chunks=chunks,
            normalizer=_normalize_punctuation,
        )
        result = self._resolve_matches(
            normalized_matches,
            "PUNCTUATION_NORMALIZED",
        )
        if result is not None:
            return result

        ordered = sorted(
            chunks,
            key=lambda item: (
                item.get("sort_order", 0),
                int(item["id"]),
            ),
        )
        normalized_quote = _normalize_punctuation(quote)
        for first, second in zip(ordered, ordered[1:]):
            combined = _normalize_punctuation(
                f"{first.get('content') or ''}{second.get('content') or ''}"
            )
            if normalized_quote and normalized_quote in combined:
                return EvidenceMappingResult(
                    status=MappingStatus.AMBIGUOUS,
                    chunk_id=None,
                    method="ADJACENT_CHUNKS",
                    reason=(
                        "证据跨越相邻Chunk，当前证据模型只能保存一个"
                        f"mapped_chunk_id：{first['id']},{second['id']}"
                    ),
                )

        return EvidenceMappingResult(
            status=MappingStatus.MISSING,
            chunk_id=None,
            method="NOT_FOUND",
            reason="在源文档的ACTIVE Chunk中未找到证据原文",
        )

    @staticmethod
    def _find_matches(
        *,
        quote: str,
        chunks: list[dict[str, Any]],
        normalizer,
    ) -> list[dict[str, Any]]:
        normalized_quote = normalizer(quote)
        if not normalized_quote:
            return []
        return [
            chunk
            for chunk in chunks
            if normalized_quote in normalizer(str(chunk.get("content") or ""))
        ]

    @staticmethod
    def _resolve_matches(
        matches: list[dict[str, Any]],
        method: str,
    ) -> EvidenceMappingResult | None:
        if len(matches) == 1:
            return EvidenceMappingResult(
                status=MappingStatus.MAPPED,
                chunk_id=int(matches[0]["id"]),
                method=method,
                reason=f"通过{method}唯一定位到Chunk",
            )
        if len(matches) > 1:
            match_ids = {int(chunk["id"]) for chunk in matches}
            parent_ids = {
                int(chunk["parent_chunk_id"])
                for chunk in matches
                if chunk.get("parent_chunk_id") is not None
                and int(chunk["parent_chunk_id"]) in match_ids
            }
            most_specific = [
                chunk
                for chunk in matches
                if int(chunk["id"]) not in parent_ids
            ]
            if parent_ids and len(most_specific) == 1:
                chunk_id = int(most_specific[0]["id"])
                return EvidenceMappingResult(
                    status=MappingStatus.MAPPED,
                    chunk_id=chunk_id,
                    method=f"{method}_MOST_SPECIFIC",
                    reason=(
                        f"证据同时命中父子Chunk，选择最具体子Chunk："
                        f"{chunk_id}"
                    ),
                )
            return EvidenceMappingResult(
                status=MappingStatus.AMBIGUOUS,
                chunk_id=None,
                method=method,
                reason=(
                    "证据同时命中多个无唯一最具体节点的Chunk："
                    f"{sorted(match_ids)}"
                ),
            )
        return None


def map_case_evidence(
    *,
    case: ReviewedEvalCase,
    source_documents: dict[str, Any],
    chunks_by_source_code: dict[str, list[dict[str, Any]]],
    mapper: EvidenceMapper | None = None,
) -> ReviewedEvalCase:
    mapper = mapper or EvidenceMapper()
    mapped_evidences = []
    for evidence in case.evidences:
        source = source_documents.get(evidence.source_doc_code)
        if source is None:
            mapped_evidences.append(
                evidence.model_copy(
                    update={
                        "mapping_status": MappingStatus.MISSING,
                        "mapping_reason": "标准证据引用的源文档不存在",
                    }
                )
            )
            continue

        result = mapper.map_quote(
            quote=evidence.evidence_quote,
            chunks=chunks_by_source_code.get(
                evidence.source_doc_code,
                [],
            ),
        )
        mapped_doc_id = _value(source, "mapped_doc_id")
        update = {
            "mapping_status": result.status,
            "mapping_reason": result.reason,
            "mapped_doc_id": (
                int(mapped_doc_id)
                if result.status == MappingStatus.MAPPED
                else None
            ),
            "mapped_chunk_id": (
                result.chunk_id
                if result.status == MappingStatus.MAPPED
                else None
            ),
        }
        mapped_evidences.append(evidence.model_copy(update=update))
    return case.model_copy(update={"evidences": mapped_evidences})


def _value(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _normalize_punctuation(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()
