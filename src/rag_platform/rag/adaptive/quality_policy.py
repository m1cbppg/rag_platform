from src.rag_platform.rag.adaptive.models import (
    RetrievalQualityDecision,
    RetrievalQualityFeatures,
    RetrievalQualityLevel,
    RetryStrategy,
)


class RetrievalQualityPolicy:
    def __init__(self, settings) -> None:
        self.settings = settings

    def decide(
        self,
        features: RetrievalQualityFeatures,
    ) -> RetrievalQualityDecision:
        score = self._score(features)

        if features.candidate_count == 0:
            return self._decision(
                level=RetrievalQualityLevel.POOR,
                score=score,
                strategy=RetryStrategy.RELAX_FILTER,
                reasons=["没有召回候选，需要放宽检索过滤条件"],
            )

        if (
            features.exact_terms
            and features.exact_term_coverage < 1.0
        ):
            missing_count = round(
                len(features.exact_terms)
                * (1.0 - features.exact_term_coverage)
            )
            return self._decision(
                level=RetrievalQualityLevel.WEAK,
                score=score,
                strategy=RetryStrategy.FORCE_BM25,
                reasons=[
                    f"候选未覆盖全部精确词，缺失约{missing_count}个"
                ],
            )

        if (
            features.comparison_intent
            and features.distinct_version_count
            < self.settings.adaptive_min_version_count
        ):
            return self._decision(
                level=RetrievalQualityLevel.WEAK,
                score=score,
                strategy=RetryStrategy.QUERY_REWRITE,
                reasons=[
                    "问题要求比较或处理版本冲突，但候选版本数量不足"
                ],
            )

        if features.target_type_coverage == 0.0:
            return self._decision(
                level=RetrievalQualityLevel.WEAK,
                score=score,
                strategy=RetryStrategy.RELAX_FILTER,
                reasons=["候选未覆盖Query分析要求的文档类型"],
            )

        if (
            features.rerank_top1
            < self.settings.adaptive_rerank_top1_threshold
            and features.rerank_top3_mean
            < self.settings.adaptive_rerank_top3_threshold
        ):
            return self._decision(
                level=RetrievalQualityLevel.WEAK,
                score=score,
                strategy=RetryStrategy.QUERY_REWRITE,
                reasons=["精排候选整体相关度偏低"],
            )

        if (
            features.candidate_count
            < self.settings.adaptive_min_candidate_count
            or features.distinct_document_count
            < self.settings.adaptive_min_distinct_documents
        ):
            return self._decision(
                level=RetrievalQualityLevel.WEAK,
                score=score,
                strategy=RetryStrategy.QUERY_REWRITE,
                reasons=["候选数量或来源文档数量不足"],
            )

        return self._decision(
            level=RetrievalQualityLevel.GOOD,
            score=score,
            strategy=RetryStrategy.NONE,
            reasons=["检索质量满足进入Context构建阶段的条件"],
        )

    def _score(
        self,
        features: RetrievalQualityFeatures,
    ) -> float:
        candidate_score = min(
            features.candidate_count
            / max(self.settings.adaptive_min_candidate_count, 1),
            1.0,
        )
        document_score = min(
            features.distinct_document_count
            / max(self.settings.adaptive_min_distinct_documents, 1),
            1.0,
        )
        version_score = (
            min(
                features.distinct_version_count
                / max(self.settings.adaptive_min_version_count, 1),
                1.0,
            )
            if features.comparison_intent
            else 1.0
        )
        score = (
            0.15 * candidate_score
            + 0.10 * document_score
            + 0.10 * min(features.channel_overlap_at_10, 1.0)
            + 0.25 * min(features.rerank_top1, 1.0)
            + 0.15 * min(features.rerank_top3_mean, 1.0)
            + 0.10 * min(features.target_type_coverage, 1.0)
            + 0.075 * min(features.exact_term_coverage, 1.0)
            + 0.075 * version_score
        )
        return round(max(0.0, min(score, 1.0)), 6)

    @staticmethod
    def _decision(
        *,
        level: RetrievalQualityLevel,
        score: float,
        strategy: RetryStrategy,
        reasons: list[str],
    ) -> RetrievalQualityDecision:
        return RetrievalQualityDecision(
            level=level,
            score=score,
            retry_strategy=strategy,
            reasons=reasons,
        )
