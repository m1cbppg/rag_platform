import json

import httpx
import pytest

from src.rag_platform.core.config import Settings
from src.rag_platform.infrastructure.dashscope_chat import DashScopeChatClient


@pytest.mark.asyncio
async def test_dashscope_chat_uses_bailian_compatible_mode_and_parses_json() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"overall_score\": 0.92}\n```"
                        }
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    settings = Settings(
        dashscope_api_key="test-key",
        dashscope_chat_base_url=(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        qwen_judge_model="qwen-plus",
    )
    client = DashScopeChatClient(settings=settings, client=http_client)

    result = await client.chat_json(
        system_prompt="审核",
        user_prompt="内容",
        temperature=0,
    )

    assert result == {"overall_score": 0.92}
    assert captured["url"].endswith("/compatible-mode/v1/chat/completions")
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "qwen-plus"
    assert captured["payload"]["temperature"] == 0
    await http_client.aclose()
