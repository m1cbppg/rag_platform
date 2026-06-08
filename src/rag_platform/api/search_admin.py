from fastapi import APIRouter, Query

from src.rag_platform.application.hybrid_search_service import HybridSearchService
from src.rag_platform.application.search_index_service import SearchIndexService
from src.rag_platform.schemas.search import (
    KeywordIndexTaskCreateResponse,
    KeywordIndexTaskRunResponse,
    SearchIndexInitResponse,
    SearchTestResponse,
)

router = APIRouter(prefix="/admin/search", tags=["search-admin"])

search_index_service = SearchIndexService()
hybrid_search_service = HybridSearchService()


@router.post("/es/index/init", response_model=SearchIndexInitResponse)
def init_es_index() -> SearchIndexInitResponse:
    """
    初始化 ES BM25 index。
    """

    return search_index_service.init_index()


@router.post("/es/tasks/create", response_model=KeywordIndexTaskCreateResponse)
def create_es_index_tasks(
    doc_id: int | None = Query(default=None, description="文档ID，不传则扫描所有chunk"),
    limit: int = Query(default=100, description="最多创建多少个任务"),
) -> KeywordIndexTaskCreateResponse:
    """
    创建 ES 索引任务。
    """

    return search_index_service.create_tasks(
        doc_id=doc_id,
        limit=limit,
    )


@router.post("/es/tasks/run", response_model=KeywordIndexTaskRunResponse)
def run_es_index_tasks(
    limit: int = Query(default=100, description="最多执行多少个任务"),
) -> KeywordIndexTaskRunResponse:
    """
    执行 ES 索引任务。
    """

    return search_index_service.run_tasks(limit=limit)


@router.get("/es/test", response_model=SearchTestResponse)
def test_bm25_search(
    q: str = Query(..., description="查询文本"),
    top_k: int = Query(default=5, description="返回数量"),
) -> SearchTestResponse:
    """
    测试 BM25 检索。
    """

    return hybrid_search_service.search_bm25(
        query=q,
        top_k=top_k,
    )


@router.get("/vector/test", response_model=SearchTestResponse)
async def test_vector_search(
    q: str = Query(..., description="查询文本"),
    top_k: int = Query(default=5, description="返回数量"),
) -> SearchTestResponse:
    """
    测试 Milvus 向量检索。
    """

    return await hybrid_search_service.search_vector(
        query=q,
        top_k=top_k,
    )


@router.get("/hybrid/test", response_model=SearchTestResponse)
async def test_hybrid_search(
    q: str = Query(..., description="查询文本"),
    top_k: int = Query(default=10, description="返回数量"),
) -> SearchTestResponse:
    """
    测试 Hybrid Search。
    """

    return await hybrid_search_service.search_hybrid(
        query=q,
        top_k=top_k,
    )