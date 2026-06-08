from fastapi import APIRouter

from src.rag_platform.api.chunks import router as chunks_router
from src.rag_platform.api.documents import router as documents_router
from src.rag_platform.api.embedding_admin import router as embedding_admin_router
from src.rag_platform.api.health import router as health_router
from src.rag_platform.api.vector_admin import router as vector_admin_router
from src.rag_platform.application.rag_service import RagService
from src.rag_platform.schemas.chat import ChatRequest, ChatResponse
from src.rag_platform.api.search_admin import router as search_admin_router
from src.rag_platform.api.retrieval import router as retrieval_router
from src.rag_platform.api.query_analysis import router as query_analysis_router
from src.rag_platform.api.rag_workflow import router as rag_workflow_router
from src.rag_platform.api.rerank_admin import router as rerank_admin_router
from src.rag_platform.api.context_admin import router as context_admin_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(documents_router)
api_router.include_router(chunks_router)
api_router.include_router(vector_admin_router)
api_router.include_router(embedding_admin_router)
api_router.include_router(search_admin_router)
api_router.include_router(retrieval_router)
api_router.include_router(query_analysis_router)
api_router.include_router(rag_workflow_router)
api_router.include_router(rerank_admin_router)
api_router.include_router(context_admin_router)


rag_service = RagService()


@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await rag_service.chat(request)