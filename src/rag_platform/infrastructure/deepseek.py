import json
from typing import Any

import httpx

from src.rag_platform.core.config import get_settings
from src.rag_platform.core.exceptions import ConfigError, ExternalServiceError


class DeepSeekClient:
    """
    DeepSeek API 客户端。

    当前先实现 query 分析需要的 JSON 调用。
    后续模块 12 会继续扩展流式回答。
    """

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.deepseek_api_key:
            raise ConfigError("DeepSeek API Key 未配置")

        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.chat_model = settings.deepseek_chat_model

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60,
        )

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        调用 DeepSeek，并要求返回 JSON。

        注意：
        即使 prompt 要求 JSON，模型仍可能输出非 JSON。
        所以这里必须做 json.loads 校验。
        """

        payload = {
            "model": self.chat_model,
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
            "temperature": 0,
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

    def _parse_json_content(self, content: str) -> dict[str, Any]:
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
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExternalServiceError(f"DeepSeek 未返回合法 JSON: {content}") from exc