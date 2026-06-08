from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.vector import VectorCollectionStatus
from src.rag_platform.infrastructure.milvus_collection import MilvusCollectionManager
from src.rag_platform.infrastructure.repositories.vector_repository import VectorRepository
from src.rag_platform.schemas.vector import VectorCollectionInitResponse


class VectorCollectionService:
    """
    向量 Collection 应用服务。

    这一层负责：
    1. 调用 Milvus 管理器初始化 collection；
    2. 把 collection 状态记录到 MySQL；
    3. 返回接口响应。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.collection_manager = MilvusCollectionManager()
        self.vector_repository = VectorRepository()

    def init_collection(self) -> VectorCollectionInitResponse:
        """
        初始化 Milvus Collection。

        这个方法可以重复执行。
        如果 collection 已存在，不会重复创建。
        """

        index_params = self.collection_manager.get_index_params()

        try:
            self.collection_manager.init_collection()

            self.vector_repository.upsert_collection_state(
                collection_name=self.settings.milvus_collection,
                embedding_model=self.settings.embedding_model,
                embedding_dimension=self.settings.embedding_dimension,
                embedding_output_type=self.settings.embedding_output_type,
                vector_field=self.settings.milvus_vector_field,
                metric_type=self.settings.milvus_metric_type,
                index_type=self.settings.milvus_index_type,
                index_params=index_params,
                status=VectorCollectionStatus.LOADED.value,
                error_message=None,
            )

            return VectorCollectionInitResponse(
                collection_name=self.settings.milvus_collection,
                embedding_model=self.settings.embedding_model,
                embedding_dimension=self.settings.embedding_dimension,
                embedding_output_type=self.settings.embedding_output_type,
                metric_type=self.settings.milvus_metric_type,
                index_type=self.settings.milvus_index_type,
                status=VectorCollectionStatus.LOADED.value,
                message="Milvus Collection 初始化完成",
            )

        except Exception as exc:
            self.vector_repository.upsert_collection_state(
                collection_name=self.settings.milvus_collection,
                embedding_model=self.settings.embedding_model,
                embedding_dimension=self.settings.embedding_dimension,
                embedding_output_type=self.settings.embedding_output_type,
                vector_field=self.settings.milvus_vector_field,
                metric_type=self.settings.milvus_metric_type,
                index_type=self.settings.milvus_index_type,
                index_params=index_params,
                status=VectorCollectionStatus.FAILED.value,
                error_message=str(exc),
            )

            raise