from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.rag_platform.core.config import get_settings


class MilvusCollectionManager:
    """
    Milvus Collection 管理器。

    当前模块只负责 collection 初始化。
    模块 5 才会真正把 embedding 向量写入 Milvus。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.collection_name = self.settings.milvus_collection
        self.vector_field = self.settings.milvus_vector_field

    def connect(self) -> None:
        """
        连接 Milvus。

        alias='default' 表示默认连接。
        后续 Collection 操作默认使用这个连接。
        """

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )

    def collection_exists(self) -> bool:
        """
        判断 collection 是否已经存在。
        """

        self.connect()
        return utility.has_collection(self.collection_name)

    def init_collection(self) -> Collection:
        """
        初始化 Milvus Collection。

        如果 collection 已存在，直接返回并 load。
        如果不存在，则创建 schema、collection、index，再 load。
        """

        self.connect()

        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            collection.load()
            return collection

        schema = self._build_schema()

        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="default",
        )

        index_params = self._build_index_params()

        collection.create_index(
            field_name=self.vector_field,
            index_params=index_params,
        )

        collection.load()

        return collection

    def _build_schema(self) -> CollectionSchema:
        """
        构建 Milvus Collection Schema。

        当前设计：
        1. chunk_id 作为 Milvus 主键；
        2. embedding 存 text-embedding-v4 生成的 dense 向量；
        3. 保留 doc_type、business_domain 等字段用于 metadata filter。
        """

        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="MySQL rag_chunk.id，作为Milvus主键",
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.INT64,
                description="MySQL rag_document.id",
            ),
            FieldSchema(
                name="chunk_code",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="chunk业务编号",
            ),
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="FAQ/SOP/RULE/MANUAL",
            ),
            FieldSchema(
                name="business_domain",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="业务域",
            ),
            FieldSchema(
                name="version",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="文档版本",
            ),
            FieldSchema(
                name="status",
                dtype=DataType.VARCHAR,
                max_length=30,
                description="ACTIVE/DISABLED",
            ),
            FieldSchema(
                name=self.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.settings.embedding_dimension,
                description="text-embedding-v4 dense embedding",
            ),
        ]

        return CollectionSchema(
            fields=fields,
            description=(
                "RAG chunk vector collection, "
                f"model={self.settings.embedding_model}, "
                f"dim={self.settings.embedding_dimension}"
            ),
            enable_dynamic_field=False,
        )

    def _build_index_params(self) -> dict:
        """
        构建 Milvus 向量索引参数。

        HNSW 参数：
        M：
            图中每个节点的最大连接数。
            越大，召回可能更好，但内存占用更高。

        efConstruction：
            构建索引时搜索候选数量。
            越大，索引质量可能更高，但构建更慢。
        """

        return {
            "index_type": self.settings.milvus_index_type,
            "metric_type": self.settings.milvus_metric_type,
            "params": {
                "M": self.settings.milvus_hnsw_m,
                "efConstruction": self.settings.milvus_hnsw_ef_construction,
            },
        }

    def get_index_params(self) -> dict:
        """
        给外部服务读取当前索引参数。
        """

        return self._build_index_params()