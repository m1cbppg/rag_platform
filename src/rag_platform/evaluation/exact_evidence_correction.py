import re
from typing import Any

from src.rag_platform.evaluation.models import (
    EvalCaseType,
    EvidenceSpec,
    MappingStatus,
    ReviewedEvalCase,
)


EXACT_EVIDENCE_CORRECTION_VERSION = "exact-identifier-v1"


def correct_exact_identifier_evidence(
    *,
    cases: list[ReviewedEvalCase],
    source_documents: list[dict[str, Any]],
    chunks_by_doc_id: dict[int, list[dict[str, Any]]],
) -> tuple[list[ReviewedEvalCase], dict[str, Any]]:
    source_by_code = {
        str(row["source_doc_code"]): row
        for row in source_documents
    }
    chunk_by_id = {
        int(chunk["id"]): chunk
        for chunks in chunks_by_doc_id.values()
        for chunk in chunks
    }
    corrected_cases: list[ReviewedEvalCase] = []
    corrections: list[dict[str, Any]] = []
    unresolved: list[str] = []
    exact_identifier_case_count = 0

    for case in cases:
        identifier = str(
            case.generation_metadata.get("required_identifier")
            or ""
        ).strip()
        if case.case_type != EvalCaseType.EXACT or not identifier:
            corrected_cases.append(case)
            continue
        exact_identifier_case_count += 1
        if _identifier_is_in_gold(
            identifier=identifier,
            evidences=case.evidences,
            chunk_by_id=chunk_by_id,
        ):
            corrected_cases.append(case)
            continue

        matches = _find_identifier_chunks(
            identifier=identifier,
            source_doc_codes=list(
                case.generation_metadata.get(
                    "source_doc_codes",
                    [],
                )
            ),
            source_by_code=source_by_code,
            chunks_by_doc_id=chunks_by_doc_id,
        )
        if not matches:
            unresolved.append(
                f"{case.case_code}:{identifier}"
            )
            corrected_cases.append(case)
            continue

        selected = matches[0]
        evidence = EvidenceSpec(
            source_doc_code=selected["source_doc_code"],
            evidence_quote=_extract_identifier_quote(
                selected["content"],
                identifier,
            ),
            fact_key=_identifier_fact_key(identifier),
            relevance_grade=3,
            mapped_doc_id=int(selected["doc_id"]),
            mapped_chunk_id=int(selected["id"]),
            mapping_status=MappingStatus.MAPPED,
            mapping_reason=(
                "EXACT评测校正：required_identifier在原Gold Chunk中缺失，"
                "通过原source_doc_codes内精确字符串匹配补充identifier证据"
            ),
        )
        evidences = [*case.evidences, evidence]
        fact_count = len(
            {
                item.fact_key
                for item in evidences
                if item.relevance_grade == 3
            }
        )
        correction_metadata = {
            "version": EXACT_EVIDENCE_CORRECTION_VERSION,
            "source_dataset_version": "v1",
            "identifier": identifier,
            "added_source_doc_code": selected["source_doc_code"],
            "added_chunk_id": int(selected["id"]),
            "candidate_chunk_ids": [
                int(item["id"])
                for item in matches
            ],
        }
        generation_metadata = {
            **case.generation_metadata,
            "evidence_correction": correction_metadata,
        }
        corrected_cases.append(
            case.model_copy(
                update={
                    "evidences": evidences,
                    "required_fact_count": max(
                        case.required_fact_count + 1,
                        fact_count,
                    ),
                    "generation_metadata": generation_metadata,
                },
                deep=True,
            )
        )
        corrections.append(
            {
                "case_code": case.case_code,
                "dataset_split": case.dataset_split.value,
                "identifier": identifier,
                "original_required_fact_count": (
                    case.required_fact_count
                ),
                "corrected_required_fact_count": max(
                    case.required_fact_count + 1,
                    fact_count,
                ),
                "added_source_doc_code": selected[
                    "source_doc_code"
                ],
                "added_chunk_id": int(selected["id"]),
                "candidate_chunk_ids": [
                    int(item["id"])
                    for item in matches
                ],
            }
        )

    if unresolved:
        raise ValueError(
            "以下EXACT题无法在声明的源文档中定位required_identifier："
            + "、".join(unresolved)
        )

    return corrected_cases, {
        "version": EXACT_EVIDENCE_CORRECTION_VERSION,
        "case_count": len(cases),
        "exact_identifier_case_count": exact_identifier_case_count,
        "corrected_case_count": len(corrections),
        "unchanged_exact_identifier_case_count": (
            exact_identifier_case_count - len(corrections)
        ),
        "corrections": corrections,
        "unresolved": [],
    }


def _identifier_is_in_gold(
    *,
    identifier: str,
    evidences: list[EvidenceSpec],
    chunk_by_id: dict[int, dict[str, Any]],
) -> bool:
    normalized = identifier.casefold()
    for evidence in evidences:
        if evidence.relevance_grade != 3:
            continue
        if normalized in evidence.evidence_quote.casefold():
            return True
        if evidence.mapped_chunk_id is None:
            continue
        chunk = chunk_by_id.get(int(evidence.mapped_chunk_id))
        if (
            chunk is not None
            and normalized
            in str(chunk.get("content") or "").casefold()
        ):
            return True
    return False


def _find_identifier_chunks(
    *,
    identifier: str,
    source_doc_codes: list[str],
    source_by_code: dict[str, dict[str, Any]],
    chunks_by_doc_id: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    normalized = identifier.casefold()
    matches = []
    for source_doc_code in source_doc_codes:
        source = source_by_code.get(source_doc_code)
        if source is None or source.get("mapped_doc_id") is None:
            continue
        doc_id = int(source["mapped_doc_id"])
        for chunk in chunks_by_doc_id.get(doc_id, []):
            content = str(chunk.get("content") or "")
            if normalized not in content.casefold():
                continue
            matches.append(
                {
                    **chunk,
                    "doc_id": doc_id,
                    "source_doc_code": source_doc_code,
                }
            )
    return sorted(
        matches,
        key=lambda item: (
            -int(
                normalized
                in str(item.get("title") or "").casefold()
            ),
            -str(item.get("content") or "").casefold().count(
                normalized
            ),
            int(item.get("sort_order") or 0),
            int(item["id"]),
        ),
    )


def _extract_identifier_quote(
    content: str,
    identifier: str,
) -> str:
    normalized = identifier.casefold()
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？；\n])", content)
        if part.strip()
    ]
    candidates = [
        part
        for part in parts
        if normalized in part.casefold()
    ]
    if not candidates:
        raise ValueError(f"Chunk正文不包含identifier：{identifier}")
    return min(candidates, key=len)


def _identifier_fact_key(identifier: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        identifier.casefold(),
    ).strip("_")
    return f"required_identifier_{normalized}"[:100]
