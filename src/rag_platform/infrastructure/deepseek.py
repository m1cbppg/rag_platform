from collections.abc import AsyncGenerator
import json
from typing import Any

import httpx

from src.rag_platform.core.config import Settings, get_settings
from src.rag_platform.core.exceptions import (
    ConfigError,
    ExternalServiceError,
    ModelResponseFormatError,
)


class DeepSeekClient:
    """
    DeepSeek API 客户端。

    当前先实现 query 分析需要的 JSON 调用。
    后续模块 12 会继续扩展流式回答。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = settings or get_settings()

        if not settings.deepseek_api_key:
            raise ConfigError("DeepSeek API Key 未配置")

        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.chat_model = settings.deepseek_chat_model
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60,
        )

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        调用 DeepSeek，并要求返回 JSON。

        注意：
        即使 prompt 要求 JSON，模型仍可能输出非 JSON。
        所以这里必须做 json.loads 校验。
        """

        payload = {
            "model": model or self.chat_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "/chat/completions",
            json=payload,
            headers=headers,
        )

        if response.status_code >= 400:
            raise ExternalServiceError(
                f"DeepSeek HTTP错误: status={response.status_code}, body={response.text}"
            )

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ExternalServiceError(f"DeepSeek 响应格式异常: {data}") from exc

        return self._parse_json_content(content)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        """
        解析模型返回的 JSON。

        有些模型可能会包一层 ```json。
        这里做简单兼容。
        """

        text = content.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()

        if text.startswith("```"):
            text = text.removeprefix("```").strip()

        if text.endswith("```"):
            text = text.removesuffix("```").strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelResponseFormatError(
                f"DeepSeek 未返回合法 JSON: {content}"
            ) from exc
        if not isinstance(result, dict):
            raise ModelResponseFormatError(
                "DeepSeek JSON 顶层结构必须是对象"
            )
        return result

    async def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        普通非流式文本生成。
        """

        payload = {
            "model": model or self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "/chat/completions",
            json=payload,
            headers=headers,
        )

        if response.status_code >= 400:
            raise ExternalServiceError(
                f"DeepSeek HTTP错误: status={response.status_code}, body={response.text}"
            )

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ExternalServiceError(f"DeepSeek 响应格式异常: {data}") from exc


    async def stream_chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        流式文本生成。

        DeepSeek Chat API 兼容 OpenAI Chat Completions。
        设置 stream=true 后，服务端会返回 SSE 格式的数据。
        """

        payload = {
            "model": model or self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with self.client.stream(
            "POST",
            "/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ExternalServiceError(
                    f"DeepSeek stream HTTP错误: status={response.status_code}, body={body.decode('utf-8')}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data_text = line.removeprefix("data:").strip()

                if data_text == "[DONE]":
                    break

                delta = self._parse_stream_delta(data_text)

                if delta:
                    yield delta

    def _parse_stream_delta(self, data_text: str) -> str:
        """
        解析 DeepSeek/OpenAI 风格 stream chunk。

        常见结构：
        {
          "choices": [
            {
              "delta": {
                "content": "xxx"
              }
            }
          ]
        }
        """

        import json

        try:
            data = json.loads(data_text)
            choices = data.get("choices") or []
            if not choices:
                return ""

            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""

        except Exception:
            return ""
