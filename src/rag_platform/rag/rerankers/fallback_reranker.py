from src.rag_platform.domain.rerank import RerankResultItem
from src.rag_platform.rag.rerankers.rerank_text_builder import RerankTextBuilder


class FallbackReranker:
    """
    Rerank 兜底器。

    失败时保留原始召回排序，避免整个 RAG 链路不可用。
    """

    def __init__(self) -> None:
        self.text_builder = RerankTextBuilder()

    def rerank(
        self,
        documents: list[dict],
        top_n: int,
    ) -> list[RerankResultItem]:
        result: list[RerankResultItem] = []

        for rank, document in enumerate(documents[:top_n], start=1):
            metadata = document.get("metadata") or {}
            chunk_id = document.get("chunk_id") or metadata.get("chunk_id")

            if chunk_id is None:
                continue

            before_score = document.get("score")

            result.append(
                RerankResultItem(
                    chunk_id=int(chunk_id),
                    document_index=rank - 1,
                    relevance_score=float(before_score or 0.0),
                    after_rank=rank,
                    text=self.text_builder.build_text(document),
                    metadata={
                        **metadata,
                        "page_content": document.get("page_content"),
                        "title": document.get("title"),
                        "title_path": document.get("title_path"),
                        "chunk_type": document.get("chunk_type"),
                        "business_domain": document.get("business_domain"),
                        "source_section": document.get("source_section"),
                        "source": document.get("source"),
                        "rerank_score": None,
                        "fallback": True,
                        "before_rank": rank,
                        "after_rank": rank,
                        "before_score": before_score,
                    },
                )
            )

        return result