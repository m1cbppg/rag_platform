import json
from typing import Any

import httpx

from src.rag_platform.core.config import Settings, get_settings
from src.rag_platform.core.exceptions import (
    ConfigError,
    ExternalServiceError,
    ModelResponseFormatError,
)


class DashScopeChatClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.dashscope_api_key:
            raise ConfigError("DashScope API Key 未配置")

        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=self.settings.dashscope_chat_base_url.rstrip("/"),
            timeout=self.settings.qwen_judge_timeout_seconds,
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
        response = await self.client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model or self.settings.qwen_judge_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code >= 400:
            raise ExternalServiceError(
                "DashScope Chat HTTP错误: "
                f"status={response.status_code}, body={response.text}"
            )

        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExternalServiceError(
                f"DashScope Chat 响应格式异常: {payload}"
            ) from exc
        return self._parse_json_content(content)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").strip()
        if text.endswith("```"):
            text = text.removesuffix("```").strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelResponseFormatError(
                f"DashScope Chat 未返回合法 JSON: {content}"
            ) from exc
        if not isinstance(result, dict):
            raise ModelResponseFormatError(
                "DashScope Chat JSON 顶层结构必须是对象"
            )
        return result
