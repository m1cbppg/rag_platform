from uuid import uuid4

from fastapi import APIRouter

from src.rag_platform.application.context_build_service import ContextBuildService
from src.rag_platform.schemas.context import (
    ContextBuildTestRequest,
    ContextBuildTestResponse,
)

router = APIRouter(prefix="/admin/context", tags=["context-admin"])


@router.post("/test", response_model=ContextBuildTestResponse)
def test_build_context(
    request: ContextBuildTestRequest,
) -> ContextBuildTestResponse:
    """
    单独测试 Context Builder。

    正式流程会在 /rag/workflow/retrieval 中自动调用。
    """

    trace_id = request.trace_id or uuid4().hex
    service = ContextBuildService()

    result, info = service.build_context(
        trace_id=trace_id,
        query_text=request.query,
        documents=request.documents,
    )

    citations = [
        {
            "citation_id": citation.citation_id,
            "chunk_id": citation.chunk_id,
            "doc_id": citation.doc_id,
            "title": citation.title,
            "title_path": citation.title_path,
            "source_section": citation.source_section,
            "chunk_type": citation.chunk_type,
            "expansion_type": citation.expansion_type,
            "sort_order": citation.sort_order,
        }
        for citation in result.citations
    ]

    return ContextBuildTestResponse(
        trace_id=trace_id,
        context=result.context,
        citations=citations,
        context_build_info=info,
    )