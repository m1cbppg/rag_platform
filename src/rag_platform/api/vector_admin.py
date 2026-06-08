from fastapi import APIRouter, Depends

from src.rag_platform.application.vector_collection_service import VectorCollectionService
from src.rag_platform.schemas.vector import VectorCollectionInitResponse

router = APIRouter(prefix="/admin/vector", tags=["vector-admin"])


def get_vector_collection_service() -> VectorCollectionService:
    return VectorCollectionService()


@router.post("/collection/init", response_model=VectorCollectionInitResponse)
def init_vector_collection(
    service: VectorCollectionService = Depends(get_vector_collection_service),
) -> VectorCollectionInitResponse:
    """
    初始化 Milvus Collection。

    注意：
    这是管理接口，不是普通用户接口。
    正式商业项目里应该加管理员权限校验。
    """

    return service.init_collection()
