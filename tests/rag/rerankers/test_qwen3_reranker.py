from src.rag_platform.rag.rerankers import qwen3_reranker as module


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict] = []

    async def rerank(self, **kwargs):
        self.calls.append(kwargs)
        return [{"index": 0, "relevance_score": 0.9}]

    async def aclose(self) -> None:
        self.closed = True


async def test_closes_rerank_http_client_after_request(
    monkeypatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(
        module,
        "DashScopeRerankClient",
        lambda: client,
    )
    reranker = module.Qwen3Reranker()

    result = await reranker.rerank(
        query="退款规则",
        documents=[
            {
                "chunk_id": 11,
                "page_content": "退款规则正文",
                "score": 0.5,
                "metadata": {},
            }
        ],
        top_n=7,
    )

    assert result[0].chunk_id == 11
    assert client.calls[0]["top_n"] == 1
    assert client.closed is True
