from fastapi import FastAPI

from src.rag_platform.api.router import api_router
from src.rag_platform.core.config import get_settings
from src.rag_platform.core.logging import setup_logging


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用。

    使用 create_app 工厂函数的好处：
    1. 方便测试；
    2. 方便后续按环境加载不同配置；
    3. 方便在启动时注册路由、中间件、异常处理器。
    """

    setup_logging()

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        version="0.1.0",
    )

    app.include_router(api_router)

    return app


app = create_app()