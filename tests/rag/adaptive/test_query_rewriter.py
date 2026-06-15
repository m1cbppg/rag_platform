from types import SimpleNamespace

from src.rag_platform.rag.adaptive.query_rewriter import QueryRewriter


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []
        self.closed = False

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        adaptive_rewrite_model="deepseek-chat",
        adaptive_rewrite_max_attempts=2,
    )


async def test_model_rewrite_returns_deduplicated_limited_queries() -> None:
    client = FakeClient(
        {
            "rewritten_query": "未出库订单 修改收货地址 条件",
            "expanded_queries": [
                "订单地址变更规则",
                "订单地址变更规则",
                "出库前修改地址",
            ],
            "reason": "补充知识库术语",
        }
    )
    rewriter = QueryRewriter(settings=_settings(), client=client)

    result = await rewriter.rewrite(
        original_question="还没出货的订单能改地址吗？",
        current_queries=["还没出货的订单能改地址吗？"],
        quality_reasons=["精排候选整体相关度偏低"],
        candidate_documents=[
            {
                "title": "订单操作",
                "page_content": "候选正文" * 100,
            }
        ],
    )

    assert result.rewritten_query == "未出库订单 修改收货地址 条件"
    assert result.expanded_queries == [
        "订单地址变更规则",
        "出库前修改地址",
    ]
    assert result.fallback_used is False
    assert len(client.calls) == 1
    assert len(client.calls[0]["user_prompt"]) < 2000


async def test_model_failure_uses_deterministic_exact_term_fallback() -> None:
    client = FakeClient(error=RuntimeError("model unavailable"))
    rewriter = QueryRewriter(settings=_settings(), client=client)

    result = await rewriter.rewrite(
        original_question="错误码 F-ORDER-001 应该怎么处理？",
        current_queries=["订单错误处理"],
        quality_reasons=["候选未覆盖全部精确词"],
        candidate_documents=[],
    )

    assert result.fallback_used is True
    assert "F-ORDER-001" in result.rewritten_query
    assert len(client.calls) == 2


async def test_invalid_model_response_uses_version_comparison_fallback() -> None:
    client = FakeClient(
        {
            "rewritten_query": "",
            "expanded_queries": "not-a-list",
            "reason": "",
        }
    )
    rewriter = QueryRewriter(settings=_settings(), client=client)

    result = await rewriter.rewrite(
        original_question="订单取消规则新旧版本有什么区别？",
        current_queries=["订单取消规则"],
        quality_reasons=["候选版本数量不足"],
        candidate_documents=[],
    )

    assert result.fallback_used is True
    assert any("旧版" in query for query in result.all_queries)
    assert any("新版" in query for query in result.all_queries)


async def test_factory_created_client_is_closed_after_rewrite() -> None:
    client = FakeClient(
        {
            "rewritten_query": "订单地址修改条件",
            "expanded_queries": [],
            "reason": "标准化表达",
        }
    )
    rewriter = QueryRewriter(
        settings=_settings(),
        client_factory=lambda: client,
    )

    await rewriter.rewrite(
        original_question="订单怎么改地址？",
        current_queries=["订单怎么改地址？"],
        quality_reasons=["精排候选整体相关度偏低"],
        candidate_documents=[],
    )

    assert client.closed is True
