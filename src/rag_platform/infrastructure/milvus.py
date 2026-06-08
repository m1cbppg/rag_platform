from pymilvus import connections

from src.rag_platform.core.config import get_settings


def connect_milvus() -> None:
    """
    连接 Milvus。

    模块 1 只负责建立连接函数。
    模块 4 会设计 Collection。
    模块 5 会真正写入向量数据。
    """

    settings = get_settings()

    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )