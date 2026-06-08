from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.search import RetrievalHit


class HybridFusion:
    """
    Hybrid Search 融合器。

    模块 10 修改：
    从“分数归一化 + 加权求和”改成 RRF。

    RRF 不直接比较 BM25 分数和向量分数，而是比较排名：
        rrf_score = Σ 1 / (rank_constant + rank)

    好处：
    1. 不依赖不同检索器分数是否同尺度；
    2. 不需要手工归一化；
    3. 对 BM25 + Vector 混合检索更稳定。
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def fuse(
        self,
        vector_hits: list[RetrievalHit],
        bm25_hits: list[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        """
        使用 RRF 融合向量召回和 BM25 召回。

        参数：
            vector_hits:
                向量召回结果，已经按向量相关性从高到低排序。

            bm25_hits:
                BM25 召回结果，已经按 ES _score 从高到低排序。

            top_k:
                最终返回数量。
        """

        final_top_k = top_k or self.settings.hybrid_final_top_k
        rank_constant = self.settings.rrf_rank_constant
        window_size = self.settings.rrf_window_size

        # 只取每一路召回的前 window_size 个结果，避免长尾噪声参与融合。
        vector_hits = vector_hits[:window_size]
        bm25_hits = bm25_hits[:window_size]

        merged: dict[int, RetrievalHit] = {}

        self._accumulate_rrf_score(
            merged=merged,
            hits=vector_hits,
            source_name="vector",
            rank_constant=rank_constant,
        )

        self._accumulate_rrf_score(
            merged=merged,
            hits=bm25_hits,
            source_name="bm25",
            rank_constant=rank_constant,
        )

        fused_hits = sorted(
            merged.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        return fused_hits[:final_top_k]

    def _accumulate_rrf_score(
        self,
        merged: dict[int, RetrievalHit],
        hits: list[RetrievalHit],
        source_name: str,
        rank_constant: int,
    ) -> None:
        """
        累加某一路召回结果的 RRF 分数。

        rank 从 1 开始。
        """

        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk_id
            rrf_score = 1.0 / (rank_constant + rank)

            if chunk_id not in merged:
                metadata = dict(hit.metadata or {})
                metadata["sources"] = [source_name]
                metadata[f"{source_name}_rank"] = rank
                metadata[f"{source_name}_raw_score"] = hit.score
                metadata["rrf_score"] = rrf_score

                merged[chunk_id] = RetrievalHit(
                    chunk_id=chunk_id,
                    score=rrf_score,
                    source="hybrid",
                    metadata=metadata,
                )
            else:
                existing = merged[chunk_id]
                existing.score += rrf_score

                existing.metadata.setdefault("sources", [])
                if source_name not in existing.metadata["sources"]:
                    existing.metadata["sources"].append(source_name)

                existing.metadata[f"{source_name}_rank"] = rank
                existing.metadata[f"{source_name}_raw_score"] = hit.score
                existing.metadata["rrf_score"] = existing.score