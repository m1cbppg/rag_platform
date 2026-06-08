import asyncio
from typing import Any

import httpx

from src.rag_platform.core.config import get_settings
from src.rag_platform.core.exceptions import ConfigError, ExternalServiceError


class DashScopeEmbeddingClient:
    """
    阿里 DashScope Embedding 客户端。

    当前使用：
    - model: text-embedding-v4
    - dimension: 1024
    - output_type: dense
    - text_type: document

    text_type 的含义：
    - document：用于底库文档向量化；
    - query：用于用户查询向量化。

    模块 5 是文档入库，所以使用 document。
    模块 7/8 做查询召回时，会用 query。
    """

    def __init__(self) -> None:
        """
        __init__ 是 Python 类的构造函数。

        当你执行：
            client = DashScopeEmbeddingClient()

        Python 会自动调用这个方法。
        """

        self.settings = get_settings()

        if not self.settings.dashscope_api_key:
            raise ConfigError("DASHSCOPE_API_KEY 未配置")

        self.base_url = self.settings.dashscope_base_url.rstrip("/")
        self.endpoint = self.settings.dashscope_embedding_endpoint
        self.model = self.settings.embedding_model
        self.dimension = self.settings.embedding_dimension
        self.output_type = self.settings.embedding_output_type
        self.timeout_seconds = self.settings.embedding_timeout_seconds
        self.max_retries = self.settings.embedding_max_retries

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文档向量。

        参数：
            texts:
                待向量化的文本列表。

        返回：
            list[list[float]]
            每个文本对应一个 float 向量。

        例如：
            [
                [0.01, 0.02, ...],
                [0.03, 0.04, ...]
            ]
        """

        if not texts:
            return []

        return await self._embed(
            texts=texts,
            text_type="document",
        )

    async def embed_query(self, text: str) -> list[float]:
        """
        生成查询向量。

        模块 5 暂时不用。
        后续检索模块会用它。
        """

        vectors = await self._embed(
            texts=[text],
            text_type="query",
        )

        return vectors[0]

    async def _embed(
        self,
        texts: list[str],
        text_type: str,
    ) -> list[list[float]]:
        """
        内部通用向量化方法。

        这里做了简单重试。
        """

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._do_embed(
                    texts=texts,
                    text_type=text_type,
                )

            except Exception as exc:
                last_error = exc

                # 如果不是最后一次，就稍微等待后重试。
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * attempt)

        raise ExternalServiceError(f"DashScope embedding 调用失败: {last_error}")

    async def _do_embed(
        self,
        texts: list[str],
        text_type: str,
    ) -> list[list[float]]:
        """
        真正发起 HTTP 请求。

        DashScope 原生接口请求体大致是：
        {
            "model": "text-embedding-v4",
            "input": {
                "texts": [...]
            },
            "parameters": {
                "dimension": 1024,
                "output_type": "dense",
                "text_type": "document"
            }
        }
        """

        payload = {
            "model": self.model,
            "input": {
                "texts": texts,
            },
            "parameters": {
                "dimension": self.dimension,
                "output_type": self.output_type,
                "text_type": text_type,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            self.endpoint,
            json=payload,
            headers=headers,
        )

        if response.status_code >= 400:
            raise ExternalServiceError(
                f"DashScope embedding HTTP错误: "
                f"status={response.status_code}, body={response.text}"
            )

        data = response.json()

        return self._parse_embeddings(data=data, expected_count=len(texts))

    def _parse_embeddings(
        self,
        data: dict[str, Any],
        expected_count: int,
    ) -> list[list[float]]:
        """
        解析 DashScope 返回的 embedding。

        原生 DashScope 返回里通常是：
        data["output"]["embeddings"]

        每个元素里有：
        {
            "embedding": [...],
            "text_index": 0
        }
        """

        output = data.get("output") or {}
        embeddings = output.get("embeddings") or []

        if len(embeddings) != expected_count:
            raise ExternalServiceError(
                f"DashScope embedding 返回数量不匹配: "
                f"expected={expected_count}, actual={len(embeddings)}"
            )

        vectors: list[list[float]] = []

        for item in embeddings:
            vector = item.get("embedding")

            if not isinstance(vector, list):
                raise ExternalServiceError("DashScope embedding 响应缺少 embedding 字段")

            if len(vector) != self.dimension:
                raise ExternalServiceError(
                    f"Embedding 维度不匹配: expected={self.dimension}, actual={len(vector)}"
                )

            vectors.append(vector)

        return vectors