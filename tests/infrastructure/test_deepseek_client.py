import json

import httpx
import pytest

from src.rag_platform.core.config import Settings
from src.rag_platform.core.exceptions import ModelResponseFormatError
from src.rag_platform.infrastructure.deepseek import DeepSeekClient


@pytest.mark.asyncio
async def test_deepseek_chat_json_accepts_generation_parameters() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "{\"source_doc_code\":\"X\"}"}}
                ]
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.deepseek.com",
    )
    client = DeepSeekClient(
        settings=Settings(
            deepseek_api_key="test-key",
            deepseek_chat_model="deepseek-chat",
        ),
        client=http_client,
    )

    result = await client.chat_json(
        system_prompt="生成",
        user_prompt="文档",
        temperature=0.4,
        max_tokens=8192,
    )

    assert result == {"source_doc_code": "X"}
    assert captured["payload"]["temperature"] == 0.4
    assert captured["payload"]["max_tokens"] == 8192
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    await http_client.aclose()


def test_deepseek_rejects_json_array_response() -> None:
    with pytest.raises(ModelResponseFormatError, match="顶层结构"):
        DeepSeekClient._parse_json_content("[]")
