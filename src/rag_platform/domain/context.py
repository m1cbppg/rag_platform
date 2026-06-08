from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ContextBuildStatus(StrEnum):
    """
    上下文构建状态。
    """

    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    TRUNCATED = "TRUNCATED"
    FAILED = "FAILED"


class ContextExpansionType(StrEnum):
    """
    chunk 进入上下文的来源类型。
    """

    SELF = "SELF"
    PARENT = "PARENT"
    PREVIOUS_NEXT = "PREVIOUS_NEXT"
    SAME_SECTION = "SAME_SECTION"


@dataclass
class ContextChunk:
    """
    进入上下文的 chunk。

    注意：
    这个对象不是数据库表对象，而是 Context Builder 内部的中间结构。
    """

    chunk_id: int
    doc_id: int | None
    content: str

    title: str | None = None
    title_path: str | None = None
    chunk_type: str | None = None
    business_domain: str | None = None
    source_section: str | None = None

    score: float | None = None
    rerank_score: float | None = None
    source: str | None = None

    expansion_type: ContextExpansionType = ContextExpansionType.SELF
    original_rank: int = 0
    sort_score: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """
    引用信息。

    citation_id：
        例如 C1、C2，后续答案里可以引用 [C1]。
    """

    citation_id: str
    chunk_id: int
    doc_id: int | None
    title: str | None
    title_path: str | None
    source_section: str | None
    chunk_type: str | None
    expansion_type: str
    sort_order: int


@dataclass
class ContextBuildResult:
    """
    Context Builder 输出结果。
    """

    context: str
    citations: list[Citation]
    used_chunks: list[ContextChunk]
    estimated_tokens: int
    status: ContextBuildStatus
    message: str