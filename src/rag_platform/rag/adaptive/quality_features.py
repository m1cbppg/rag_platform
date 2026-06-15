import re

from src.rag_platform.rag.adaptive.models import (
    RetrievalQualityFeatures,
)


_EXACT_TERM_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]*"
        r"(?:-[A-Z0-9]+)+(?![A-Za-z0-9])"
    ),
    re.compile(r"(?<![A-Za-z0-9])[A-Z]\d{3,}(?![A-Za-z0-9])"),
    re.compile(
        r"(?<![A-Za-z0-9])[a-zA-Z][a-zA-Z0-9]*"
        r"(?:_[a-zA-Z0-9]+)+(?![A-Za-z0-9])"
    ),
    re.compile(r"【[^】]+】"),
    re.compile(r"/api/[a-zA-Z0-9_/\-]+"),
)

_COMPARISON_TERMS = (
    "新旧",
    "新版",
    "旧版",
    "版本",
    "不同",
    "区别",
    "变更",
    "冲突",
    "优先",
    "生效",
)

_EXPLICIT_VERSION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[Vv]\s*(\d+(?:\.\d+)?)(?![A-Za-z0-9])"
)


def extract_exact_terms(question: str) -> list[str]:
    matches: list[str] = []
    for pattern in _EXACT_TERM_PATTERNS:
        matches.extend(pattern.findall(question or ""))
    return list(dict.fromkeys(matches))


def has_comparison_intent(question: str) -> bool:
    return (
        any(term in question for term in _COMPARISON_TERMS)
        or len(_EXPLICIT_VERSION_PATTERN.findall(question)) >= 2
    )


def extract_retrieval_quality_features(
    *,
    question: str,
    documents: list[dict],
    reranked_documents: list[dict],
    target_doc_types: list[str] | None = None,
) -> RetrievalQualityFeatures:
    top_documents = documents[:10]
    document_ids = {
        _value(document, "doc_id")
        for document in documents
        if _value(document, "doc_id") is not None
    }
    overlap_count = sum(
        1
        for document in top_documents
        if {"bm25", "vector"}.issubset(
            {
                str(source).lower()
                for source in (
                    _metadata(document).get("sources") or []
                )
            }
        )
    )
    overlap = (
        overlap_count / len(top_documents)
        if top_documents
        else 0.0
    )

    rerank_scores = [
        _score(document)
        for document in reranked_documents
    ]
    top1 = rerank_scores[0] if rerank_scores else 0.0
    top3_scores = rerank_scores[:3]
    top3_mean = (
        sum(top3_scores) / len(top3_scores)
        if top3_scores
        else 0.0
    )
    margin = (
        max(top1 - rerank_scores[1], 0.0)
        if len(rerank_scores) > 1
        else top1
    )

    expected_types = {
        str(value).upper()
        for value in (target_doc_types or [])
        if value
    }
    actual_types = {
        str(_value(document, "chunk_type")).upper()
        for document in documents
        if _value(document, "chunk_type")
    }
    type_coverage = (
        len(expected_types & actual_types) / len(expected_types)
        if expected_types
        else 1.0
    )

    exact_terms = extract_exact_terms(question)
    exact_evidence_documents = (
        reranked_documents
        if reranked_documents
        else documents
    )
    searchable_text = "\n".join(
        _document_search_text(document)
        for document in exact_evidence_documents
    ).casefold()
    covered_exact_terms = [
        term
        for term in exact_terms
        if term.casefold() in searchable_text
    ]
    exact_coverage = (
        len(covered_exact_terms) / len(exact_terms)
        if exact_terms
        else 1.0
    )

    version_evidence_documents = (
        reranked_documents
        if reranked_documents
        else documents
    )
    versions = _explicit_versions(version_evidence_documents)
    comparison_intent = has_comparison_intent(question)

    return RetrievalQualityFeatures(
        candidate_count=len(documents),
        distinct_document_count=len(document_ids),
        channel_overlap_at_10=round(overlap, 6),
        rerank_top1=round(top1, 6),
        rerank_top3_mean=round(top3_mean, 6),
        rerank_margin=round(margin, 6),
        target_type_coverage=round(type_coverage, 6),
        exact_terms=exact_terms,
        exact_term_coverage=round(exact_coverage, 6),
        distinct_version_count=len(versions),
        comparison_intent=comparison_intent,
    )


def _metadata(document: dict) -> dict:
    return document.get("metadata") or {}


def _value(document: dict, key: str):
    value = document.get(key)
    if value is not None:
        return value
    return _metadata(document).get(key)


def _score(document: dict) -> float:
    value = (
        document.get("rerank_score")
        or _metadata(document).get("rerank_score")
        or document.get("score")
        or 0.0
    )
    return float(value)


def _document_search_text(document: dict) -> str:
    metadata = _metadata(document)
    values = [
        document.get("page_content"),
        document.get("title"),
        document.get("title_path"),
        document.get("source_section"),
        metadata.get("title"),
        metadata.get("title_path"),
        metadata.get("source_section"),
        metadata.get("chunk_code"),
    ]
    return "\n".join(
        str(value)
        for value in values
        if value
    )


def _explicit_versions(documents: list[dict]) -> set[str]:
    versions: set[str] = set()
    for document in documents:
        text = _version_identity_text(document)
        matches = _EXPLICIT_VERSION_PATTERN.findall(text)
        versions.update(
            _normalize_version(value)
            for value in matches
        )
        if matches:
            continue
        metadata_version = str(
            _value(document, "version") or ""
        ).strip()
        if metadata_version[:1].casefold() == "v":
            versions.add(
                _normalize_version(metadata_version[1:])
            )
    return {value for value in versions if value}


def _version_identity_text(document: dict) -> str:
    content = str(document.get("page_content") or "")
    first_line = content.splitlines()[0] if content else ""
    return first_line


def _normalize_version(value: str) -> str:
    normalized = value.strip().casefold()
    try:
        return str(float(normalized)).removesuffix(".0")
    except ValueError:
        return normalized
