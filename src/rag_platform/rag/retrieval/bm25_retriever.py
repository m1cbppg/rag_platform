from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.search import RetrievalHit
from src.rag_platform.infrastructure.elasticsearch_store import ElasticsearchChunkStore


class BM25Retriever:
    """
    ES BM25 检索器。

    后续模块 7 会把它进一步封装成 LangChain Retriever。
    当前先保持普通 Python 类，便于理解底层逻辑。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ElasticsearchChunkStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
        business_domain: str | None = None,
    ) -> list[RetrievalHit]:
        actual_top_k = top_k or self.settings.es_bm25_top_k

        raw_hits = self.store.search_bm25(
            query=query,
            top_k=actual_top_k,
            doc_type=doc_type,
            business_domain=business_domain,
        )

        return [
            RetrievalHit(
                chunk_id=item["chunk_id"],
                score=item["score"],
                source="bm25",
                metadata=item["metadata"],
            )
            for item in raw_hits
        ]