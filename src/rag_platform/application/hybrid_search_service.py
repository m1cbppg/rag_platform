from src.rag_platform.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag_platform.rag.retrieval.hybrid_fusion import HybridFusion
from src.rag_platform.rag.retrieval.vector_retriever import VectorRetriever
from src.rag_platform.schemas.search import SearchHitResponse, SearchTestResponse


class HybridSearchService:
    """
    Hybrid Search 应用服务。

    当前用于测试：
    1. BM25 单独检索；
    2. Vector 单独检索；
    3. Hybrid 合并检索。

    后续模块 7 会封装成 LangChain Retriever。
    """

    def __init__(self) -> None:
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.fusion = HybridFusion()

    def search_bm25(
        self,
        query: str,
        top_k: int,
    ) -> SearchTestResponse:
        hits = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return self._to_response(query, hits)

    async def search_vector(
        self,
        query: str,
        top_k: int,
    ) -> SearchTestResponse:
        hits = await self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return self._to_response(query, hits)

    async def search_hybrid(
        self,
        query: str,
        top_k: int,
    ) -> SearchTestResponse:
        vector_hits = await self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        bm25_hits = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        hybrid_hits = self.fusion.fuse(
            vector_hits=vector_hits,
            bm25_hits=bm25_hits,
            top_k=top_k,
        )

        return self._to_response(query, hybrid_hits)

    def _to_response(
        self,
        query: str,
        hits,
    ) -> SearchTestResponse:
        return SearchTestResponse(
            query=query,
            hits=[
                SearchHitResponse(
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    source=hit.source,
                    title=hit.metadata.get("title"),
                    title_path=hit.metadata.get("title_path"),
                    content=hit.metadata.get("content"),
                )
                for hit in hits
            ],
        )