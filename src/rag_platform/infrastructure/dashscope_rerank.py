import asyncio
import logging
from typing import Any

import httpx

from src.rag_platform.core.config import get_settings
from src.rag_platform.core.exceptions import ConfigError, ExternalServiceError

logger = logging.getLogger(__name__)
class DashScopeRerankClient:
    """
    百炼 qwen3-rerank HTTP 客户端。

    注意：
    qwen3-rerank 使用的是：
        POST /compatible-api/v1/reranks

    请求体是扁平结构：
        {
            "model": "qwen3-rerank",
            "documents": [...],
            "query": "...",
            "top_n": 5,
            "instruct": "..."
        }

    不要写成 embedding 那种 input / parameters 结构。
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.dashscope_api_key:
            raise ConfigError("DASHSCOPE_API_KEY 未配置，无法调用 qwen3-rerank")

        self.base_url = self.settings.dashscope_rerank_base_url.rstrip("/")
        self.endpoint = self.settings.dashscope_rerank_endpoint
        self.model = self.settings.rerank_model
        self.timeout_seconds = self.settings.rerank_timeout_seconds
        self.max_retries = self.settings.rerank_max_retries

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        instruct: str,
    ) -> list[dict[str, Any]]:
        """
        调用 qwen3-rerank。

        返回：
        [
            {
                "index": 0,
                "relevance_score": 0.93
            }
        ]
        """

        if not documents:
            return []

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._do_rerank(
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    instruct=instruct,
                )
            except Exception as exc:
                last_error = exc

                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * attempt)

        raise ExternalServiceError(f"qwen3-rerank 调用失败: {last_error}")

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _do_rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
        instruct: str,
    ) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "documents": documents,
            "query": query,
            "top_n": top_n,
            "instruct": instruct,
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
                f"qwen3-rerank HTTP错误: "
                f"status={response.status_code}, body={response.text}"
            )

        data = response.json()
        logger.info("rerank result: %s", data)
        return self._parse_response(data)

    def _parse_response(
            self,
            data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        解析 qwen3-rerank 返回。

        qwen3-rerank compatible endpoint 可能返回：
        1. {"results": [...]}
        2. {"output": {"results": [...]}}

        所以这里两个位置都兼容。
        """

        results = data.get("results")

        if results is None:
            output = data.get("output") or {}
            results = output.get("results")

        if results is None:
            raise ExternalServiceError(
                f"qwen3-rerank 响应中没有 results 字段: {data}"
            )

        if not isinstance(results, list):
            raise ExternalServiceError(
                f"qwen3-rerank results 不是 list: {data}"
            )

        parsed: list[dict[str, Any]] = []

        for item in results:
            if "index" not in item:
                raise ExternalServiceError(
                    f"qwen3-rerank 响应缺少 index: {data}"
                )

            score = (
                item.get("relevance_score")
                if "relevance_score" in item
                else item.get("score")
            )

            if score is None:
                raise ExternalServiceError(
                    f"qwen3-rerank 响应缺少 relevance_score/score: {data}"
                )

            parsed.append({
                "index": int(item["index"]),
                "relevance_score": float(score),
            })

        return parsed
