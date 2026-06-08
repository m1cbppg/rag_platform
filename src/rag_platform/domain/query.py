from enum import StrEnum


class RetrievalMode(StrEnum):
    """
    检索模式。

    BM25：
        只走 Elasticsearch BM25。
        适合条款编号、按钮名、字段名、错误码等精确匹配。

    VECTOR：
        只走 Milvus 向量检索。
        适合口语化、语义相似表达。

    HYBRID：
        ES BM25 + Milvus 向量召回。
        默认推荐，用于大多数业务问答。
    """

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


class QueryIntent(StrEnum):
    """
    用户问题意图。

    FAQ：
        常见问题问答。

    PROCEDURE：
        流程排查类问题，对应 SOP。

    RULE:
        规则判断类问题，对应业务规则。

    MANUAL:
        后台操作类问题，对应操作手册。

    UNKNOWN:
        无法明确判断。
    """

    FAQ = "faq"
    PROCEDURE = "procedure"
    RULE = "rule"
    MANUAL = "manual"
    UNKNOWN = "unknown"