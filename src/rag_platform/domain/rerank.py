from dataclasses import dataclass
from enum import StrEnum


class RerankStatus(StrEnum):
    """
    Rerank 执行状态。
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    FALLBACK = "FALLBACK"
    SKIPPED = "SKIPPED"


@dataclass
class RerankCandidate:
    """
    Rerank 输入候选。

    document_index：
        当前候选在 documents 列表中的原始位置。
        qwen3-rerank 返回 index 时，需要用它映射回原始 chunk。
    """

    document_index: int
    chunk_id: int
    text: str
    metadata: dict
    before_rank: int
    before_score: float | None


@dataclass
class RerankResultItem:
    """
    Rerank 输出结果。
    """

    chunk_id: int
    document_index: int
    relevance_score: float
    after_rank: int
    text: str
    metadata: dict