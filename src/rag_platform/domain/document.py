from enum import StrEnum


class DocumentType(StrEnum):
    """
    文档类型枚举。
    """

    FAQ = "FAQ"
    SOP = "SOP"
    RULE = "RULE"
    MANUAL = "MANUAL"


class DocumentStatus(StrEnum):
    """
    文档处理状态枚举。
    """

    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    CLEANED = "CLEANED"

    # 模块 3 新增：
    # 表示文档已经完成 chunk 切分。
    CHUNKED = "CHUNKED"

    NEED_REVIEW = "NEED_REVIEW"
    FAILED = "FAILED"