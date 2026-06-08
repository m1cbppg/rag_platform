from dataclasses import dataclass
from enum import StrEnum


class ChunkType(StrEnum):
    """
    chunk 类型枚举。

    当前和 DocumentType 保持一致。
    后续也可以扩展：
    - FAQ_QUESTION
    - FAQ_ANSWER
    - SOP_PARENT
    - SOP_CHILD
    """

    FAQ = "FAQ"
    SOP = "SOP"
    RULE = "RULE"
    MANUAL = "MANUAL"


class ChunkRelationType(StrEnum):
    """
    chunk 关系类型。

    PARENT_CHILD：
        父子关系。

    PREVIOUS_NEXT：
        相邻关系。

    SAME_SECTION：
        同章节关系。
    """

    PARENT_CHILD = "PARENT_CHILD"
    PREVIOUS_NEXT = "PREVIOUS_NEXT"
    SAME_SECTION = "SAME_SECTION"


@dataclass
class ChunkBuildItem:
    """
    chunk 构建结果。

    chunker 不直接写数据库。
    它只负责把 structure_json 转成 ChunkBuildItem。

    后续由 Application Service 和 Repository 负责入库。
    """

    chunk_type: ChunkType
    title: str | None
    title_path: str | None
    content: str
    summary: str | None = None
    keywords: str | None = None
    tags: str | None = None
    source_section: str | None = None
    parent_temp_key: str | None = None
    temp_key: str | None = None
    sort_order: int = 0