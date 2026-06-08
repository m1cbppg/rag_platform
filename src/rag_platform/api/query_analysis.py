from fastapi import APIRouter

from src.rag_platform.application.query_understanding_service import QueryUnderstandingService
from src.rag_platform.schemas.query_analysis import (
    QueryAnalysisRequest,
    QueryAnalysisResponse,
)

router = APIRouter(prefix="/query", tags=["query-understanding"])


@router.post("/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(
    request: QueryAnalysisRequest,
) -> QueryAnalysisResponse:
    """
    Query 理解接口。

    只分析问题，不执行检索。
    适合调试：
    1. query rewrite 是否合理；
    2. target_doc_types 是否正确；
    3. retrieval_mode 是否正确；
    4. 是否触发兜底。
    """

    service = QueryUnderstandingService()

    return await service.analyze(request)