class RagPlatformError(Exception):
    """
    RAG 平台基础异常。

    所有自定义异常都可以继承它。
    这样后续 FastAPI 可以统一捕获 RagPlatformError。
    """


class ConfigError(RagPlatformError):
    """
    配置错误。

    例如：
    - DeepSeek API Key 没配置；
    - MySQL 地址为空；
    - Milvus collection 未配置。
    """


class ExternalServiceError(RagPlatformError):
    """
    外部服务调用错误。

    例如：
    - MySQL 连接失败；
    - Milvus 查询失败；
    - DeepSeek API 超时。
    """