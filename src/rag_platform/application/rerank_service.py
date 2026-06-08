import time

from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.rerank import RerankResultItem, RerankStatus
from src.rag_platform.infrastructure.repositories.rerank_repository import RerankRepository
from src.rag_platform.rag.rerankers.fallback_reranker import FallbackReranker
from src.rag_platform.rag.rerankers.qwen3_reranker import Qwen3Reranker


class RerankService:
    """
    Rerank 应用服务。

    工程化职责：
    1. 控制是否启用 rerank；
    2. 限制候选数量；
    3. 调用 qwen3-rerank；
    4. 失败时按配置 fail-open；
    5. 保存 rerank 日志；
    6. 输出统一 reranked_documents。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.repository = RerankRepository()
        self.fallback_reranker = FallbackReranker()

    async def rerank_documents(
        self,
        trace_id: str,
        query: str,
        documents: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        对 merged_documents 进行 rerank。

        返回：
            reranked_documents, rerank_info
        """

        if not self.settings.rerank_enabled:
            items = self.fallback_reranker.rerank(
                documents=documents,
                top_n=self.settings.rerank_top_n,
            )

            log_id = self.repository.create_rerank_log(
                trace_id=trace_id,
                query_text=query,
                provider=self.settings.rerank_provider,
                model=self.settings.rerank_model,
                candidate_count=len(documents),
                top_n=self.settings.rerank_top_n,
                success_count=len(items),
                status=RerankStatus.SKIPPED.value,
                fail_open=True,
                latency_ms=0,
                error_message="Rerank disabled",
            )
            self.repository.save_rerank_items(log_id, trace_id, items)

            return self._items_to_documents(items), {
                "status": RerankStatus.SKIPPED.value,
                "rerank_log_id": log_id,
                "message": "Rerank disabled, fallback to original order",
            }

        start_time = time.perf_counter()

        try:
            reranker = Qwen3Reranker()

            items = await reranker.rerank(
                query=query,
                documents=documents,
            )

            items = self._filter_by_min_score(items)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            log_id = self.repository.create_rerank_log(
                trace_id=trace_id,
                query_text=query,
                provider=self.settings.rerank_provider,
                model=self.settings.rerank_model,
                candidate_count=len(documents),
                top_n=self.settings.rerank_top_n,
                success_count=len(items),
                status=RerankStatus.SUCCESS.value,
                fail_open=False,
                latency_ms=latency_ms,
                error_message=None,
            )
            self.repository.save_rerank_items(log_id, trace_id, items)

            return self._items_to_documents(items), {
                "status": RerankStatus.SUCCESS.value,
                "rerank_log_id": log_id,
                "latency_ms": latency_ms,
                "candidate_count": len(documents),
                "success_count": len(items),
            }

        except Exception as exc:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if not self.settings.rerank_fail_open:
                log_id = self.repository.create_rerank_log(
                    trace_id=trace_id,
                    query_text=query,
                    provider=self.settings.rerank_provider,
                    model=self.settings.rerank_model,
                    candidate_count=len(documents),
                    top_n=self.settings.rerank_top_n,
                    success_count=0,
                    status=RerankStatus.FAILED.value,
                    fail_open=False,
                    latency_ms=latency_ms,
                    error_message=str(exc),
                )

                raise

            items = self.fallback_reranker.rerank(
                documents=documents,
                top_n=self.settings.rerank_top_n,
            )

            log_id = self.repository.create_rerank_log(
                trace_id=trace_id,
                query_text=query,
                provider=self.settings.rerank_provider,
                model=self.settings.rerank_model,
                candidate_count=len(documents),
                top_n=self.settings.rerank_top_n,
                success_count=len(items),
                status=RerankStatus.FALLBACK.value,
                fail_open=True,
                latency_ms=latency_ms,
                error_message=str(exc),
            )
            self.repository.save_rerank_items(log_id, trace_id, items)

            return self._items_to_documents(items), {
                "status": RerankStatus.FALLBACK.value,
                "rerank_log_id": log_id,
                "latency_ms": latency_ms,
                "error_message": str(exc),
            }

    def _filter_by_min_score(
        self,
        items: list[RerankResultItem],
    ) -> list[RerankResultItem]:
        """
        根据 rerank_min_score 过滤弱相关候选。

        默认 0.0，等于不过滤。
        """
        min_score = self.settings.rerank_min_score

        return [
            item
            for item in items
            if item.relevance_score >= min_score
        ]

    def _items_to_documents(
        self,
        items: list[RerankResultItem],
    ) -> list[dict]:
        """
        RerankResultItem 转回 LangGraph State 可保存的 dict。
        """

        documents: list[dict] = []

        for item in items:
            metadata = item.metadata or {}
            page_content = metadata.get("page_content") or item.text

            documents.append({
                "page_content": page_content,
                "metadata": {
                    **metadata,
                    "chunk_id": item.chunk_id,
                    "rerank_score": item.relevance_score,
                    "after_rank": item.after_rank,
                },
                "chunk_id": item.chunk_id,
                "score": item.relevance_score,
                "source": metadata.get("source"),
                "title": metadata.get("title"),
                "title_path": metadata.get("title_path"),
                "chunk_type": metadata.get("chunk_type"),
                "business_domain": metadata.get("business_domain"),
                "source_section": metadata.get("source_section"),
                "rerank_score": item.relevance_score,
                "after_rank": item.after_rank,
            })

        return documents