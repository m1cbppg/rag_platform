from langchain_core.documents import Document

from src.rag_platform.application.query_understanding_service import QueryUnderstandingService
from src.rag_platform.rag.retrievers.langchain_bm25_retriever import LangChainBM25Retriever
from src.rag_platform.rag.retrievers.langchain_hybrid_retriever import LangChainHybridRetriever
from src.rag_platform.rag.retrievers.langchain_vector_retriever import LangChainVectorRetriever
from src.rag_platform.schemas.query_analysis import QueryAnalysisRequest
from src.rag_platform.schemas.retrieval import (
    RetrievedDocumentResponse,
    RetrievalRequest,
    RetrievalResponse,
)


class RetrievalService:
    """
    检索应用服务。

    模块 8 增强：
    1. 支持 mode=auto；
    2. auto 时先做 Query 理解；
    3. 根据 QueryAnalysisResult 选择 retriever。
    """

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        analysis_result = None
        actual_request = request

        if request.mode.lower() == "auto":
            understanding_service = QueryUnderstandingService()

            analysis_response = await understanding_service.analyze(
                QueryAnalysisRequest(
                    query=request.query,
                    business_domain=request.business_domain,
                )
            )

            analysis_result = analysis_response.result

            actual_request = RetrievalRequest(
                query=analysis_result.rewritten_query,
                mode=analysis_result.retrieval_mode,
                top_k=request.top_k,
                doc_type=(
                    analysis_result.target_doc_types[0]
                    if len(analysis_result.target_doc_types) == 1
                    else request.doc_type
                ),
                business_domain=analysis_result.business_domain or request.business_domain,
            )

        retriever = self._build_retriever(actual_request)

        documents = await retriever.ainvoke(actual_request.query)

        response = RetrievalResponse(
            query=request.query,
            mode=actual_request.mode,
            documents=[
                self._to_response_document(document)
                for document in documents
            ],
        )

        return response

    def _build_retriever(self, request: RetrievalRequest):
        mode = request.mode.lower()

        if mode == "bm25":
            return LangChainBM25Retriever(
                top_k=request.top_k,
                doc_type=request.doc_type,
                business_domain=request.business_domain,
            )

        if mode == "vector":
            return LangChainVectorRetriever(
                top_k=request.top_k,
                doc_type=request.doc_type,
                business_domain=request.business_domain,
            )

        if mode == "hybrid":
            return LangChainHybridRetriever(
                top_k=request.top_k,
                vector_top_k=request.top_k,
                bm25_top_k=request.top_k,
                doc_type=request.doc_type,
                business_domain=request.business_domain,
            )

        raise ValueError(f"不支持的检索模式: {request.mode}")

    def _to_response_document(
        self,
        document: Document,
    ) -> RetrievedDocumentResponse:
        metadata = document.metadata or {}

        return RetrievedDocumentResponse(
            chunk_id=int(metadata.get("chunk_id")),
            score=metadata.get("score"),
            source=metadata.get("source"),
            title=metadata.get("title"),
            title_path=metadata.get("title_path"),
            chunk_type=metadata.get("chunk_type"),
            business_domain=metadata.get("business_domain"),
            source_section=metadata.get("source_section"),
            content=document.page_content,
        )