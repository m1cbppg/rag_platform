from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class KeywordIndexTaskStatus(StrEnum):
    """
    ES 关键词索引任务状态。

    PENDING：
        等待写入 ES。

    PROCESSING：
        正在写入 ES。

    SUCCESS：
        已成功写入 ES。

    FAILED：
        写入失败，可重试。

    SKIPPED：
        跳过，例如索引文本未变化。
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class RetrievalHit:
    """
    统一召回结果对象。

    不管是 ES BM25 召回，还是 Milvus 向量召回，最后都转成这个结构。

    chunk_id：
        MySQL rag_chunk.id。

    score：
        当前召回通道的原始分数。

    source：
        召回来源，例如 bm25 / vector / hybrid。

    metadata：
        额外信息，例如 doc_type、title_path、business_domain。
    """

    chunk_id: int
    score: float
    source: str
    metadata: dict[str, Any]