from fastapi import APIRouter

from src.rag_platform.core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict:
    """
    健康检查接口。

    GET /health

    返回服务名称和当前环境。
    """

    settings = get_settings()

    return {
        "status": "ok",
        "app_name": settings.app_name,
        "env": settings.app_env,
    }