from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.rerank import RerankCandidate, RerankResultItem
from src.rag_platform.infrastructure.dashscope_rerank import DashScopeRerankClient
from src.rag_platform.rag.rerankers.rerank_text_builder import RerankTextBuilder


class Qwen3Reranker:
    """
    qwen3-rerank 精排器。

    职责：
    1. 把候选 document 转成 rerank 文本；
    2. 调用百炼 qwen3-rerank；
    3. 根据返回 index 映射回原始候选；
    4. 生成排序后的 RerankResultItem。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = DashScopeRerankClient()
        self.text_builder = RerankTextBuilder()

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int | None = None,
    ) -> list[RerankResultItem]:
        """
        对候选文档做 rerank。

        documents:
            来自 LangGraph State 的 merged_documents。
        """

        try:
            candidate_limit = self.settings.rerank_candidate_limit
            effective_top_n = max(
                1,
                int(top_n or self.settings.rerank_top_n),
            )

            limited_documents = documents[:candidate_limit]

            candidates = self._build_candidates(limited_documents)

            if not candidates:
                return []

            rerank_results = await self.client.rerank(
                query=query,
                documents=[
                    candidate.text
                    for candidate in candidates
                ],
                top_n=min(effective_top_n, len(candidates)),
                instruct=self.settings.rerank_instruct,
            )

            candidate_map = {
                candidate.document_index: candidate
                for candidate in candidates
            }

            result_items: list[RerankResultItem] = []

            for after_rank, item in enumerate(
                rerank_results,
                start=1,
            ):
                original_index = item["index"]
                candidate = candidate_map.get(original_index)

                if candidate is None:
                    continue

                result_items.append(
                    RerankResultItem(
                        chunk_id=candidate.chunk_id,
                        document_index=original_index,
                        relevance_score=item["relevance_score"],
                        after_rank=after_rank,
                        text=candidate.text,
                        metadata={
                            **candidate.metadata,
                            "rerank_score": item[
                                "relevance_score"
                            ],
                            "before_rank": candidate.before_rank,
                            "after_rank": after_rank,
                            "before_score": (
                                candidate.before_score
                            ),
                        },
                    )
                )

            return result_items
        finally:
            close = getattr(self.client, "aclose", None)
            if close is not None:
                await close()

    def _build_candidates(
        self,
        documents: list[dict],
    ) -> list[RerankCandidate]:
        candidates: list[RerankCandidate] = []

        for index, document in enumerate(documents):
            metadata = document.get("metadata") or {}

            chunk_id = document.get("chunk_id") or metadata.get("chunk_id")
            if chunk_id is None:
                continue

            text = self.text_builder.build_text(document)

            if not text:
                continue

            candidates.append(
                RerankCandidate(
                    document_index=index,
                    chunk_id=int(chunk_id),
                    text=text,
                    metadata={
                        **metadata,
                        "page_content": document.get("page_content"),
                        "title": document.get("title"),
                        "title_path": document.get("title_path"),
                        "chunk_type": document.get("chunk_type"),
                        "business_domain": document.get("business_domain"),
                        "source_section": document.get("source_section"),
                        "source": document.get("source"),
                    },
                    before_rank=index + 1,
                    before_score=document.get("score"),
                )
            )

        return candidates
