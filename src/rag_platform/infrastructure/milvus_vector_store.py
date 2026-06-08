from pymilvus import Collection, connections

from src.rag_platform.core.config import get_settings


class MilvusVectorStore:
    """
    Milvus 向量写入封装。

    当前只负责插入 / 覆盖向量。
    检索会在后续 Retriever 模块实现。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.collection_name = self.settings.milvus_collection

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )

        self.collection = Collection(self.collection_name)

    def upsert_vectors(self, rows: list[dict]) -> None:
        """
        批量写入向量。

        rows 中每个元素格式：
        {
            "chunk_id": 1,
            "doc_id": 1,
            "chunk_code": "CHK_xxx",
            "doc_type": "FAQ",
            "business_domain": "客服流程",
            "version": "2026-06",
            "status": "ACTIVE",
            "embedding": [...]
        }

        注意：
        Milvus 不同版本对 upsert 支持情况不同。
        为了学习和兼容，这里采用 delete + insert。
        """

        if not rows:
            return

        chunk_ids = [row["chunk_id"] for row in rows]

        self._delete_existing(chunk_ids)

        data = [
            [row["chunk_id"] for row in rows],
            [row["doc_id"] for row in rows],
            [row["chunk_code"] for row in rows],
            [row["doc_type"] for row in rows],
            [row.get("business_domain") or "" for row in rows],
            [row.get("version") or "" for row in rows],
            [row.get("status") or "ACTIVE" for row in rows],
            [row["embedding"] for row in rows],
        ]

        self.collection.insert(data)
        self.collection.flush()

    def _delete_existing(self, chunk_ids: list[int]) -> None:
        """
        删除已有向量。

        这样重复执行 embedding 入库时，不会产生重复向量。
        """

        if not chunk_ids:
            return

        id_text = ", ".join(str(x) for x in chunk_ids)
        expr = f"chunk_id in [{id_text}]"

        self.collection.delete(expr)
        self.collection.flush()