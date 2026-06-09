import hashlib
import json
from pathlib import Path

from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    GeneratedSourceDocument,
    ReviewedDocumentOutcome,
)
from src.rag_platform.evaluation.models import ReviewStatus, SourceDocumentSpec


def build_source_document_spec(
    *,
    blueprint: DocumentBlueprint,
    document: GeneratedSourceDocument,
    source_path: Path,
    project_root: Path,
    outcome: ReviewedDocumentOutcome | None = None,
) -> SourceDocumentSpec:
    source_bytes = source_path.read_bytes()
    review_status = ReviewStatus.PENDING
    review_score: float | None = None
    review_reason: str | None = None
    review_history: list[dict] = []

    if outcome is not None:
        review_status = (
            ReviewStatus.PASSED
            if outcome.review.passed
            else ReviewStatus.REJECTED
        )
        review_score = outcome.review.overall_score
        review_reason = json.dumps(
            {
                "summary": outcome.review.summary,
                "issues": outcome.review.issues,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        review_history = [
            item.model_dump(mode="json")
            for item in outcome.history
        ]

    try:
        relative_path = source_path.resolve().relative_to(
            project_root.resolve()
        )
    except ValueError:
        relative_path = Path(source_path.name)

    return SourceDocumentSpec(
        source_doc_code=document.source_doc_code,
        title=document.title,
        doc_type=document.doc_type,
        topic=document.topic,
        version=document.version,
        effective_from=document.effective_from,
        effective_to=document.effective_to,
        is_current=blueprint.is_current,
        relative_file_path=relative_path.as_posix(),
        source_content_sha256=hashlib.sha256(source_bytes).hexdigest(),
        generation_spec={
            "blueprint": blueprint.model_dump(mode="json"),
            "review_history": review_history,
        },
        review_status=review_status,
        review_score=review_score,
        review_reason=review_reason,
    )
