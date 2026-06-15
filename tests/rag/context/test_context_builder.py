from types import SimpleNamespace

from src.rag_platform.rag.context.context_builder import ContextBuilder


class NoopExpander:
    def expand(self, chunks):
        return chunks


def make_builder() -> ContextBuilder:
    builder = ContextBuilder()
    builder.settings = SimpleNamespace(
        context_max_chunks=10,
        context_max_tokens=4000,
        context_include_metadata_header=False,
    )
    builder.expander = NoopExpander()
    return builder


def test_decomposed_context_groups_evidence_by_sub_query() -> None:
    builder = make_builder()
    documents = [
        {
            "chunk_id": 1,
            "page_content": "未出库订单可以修改收货地址。",
            "score": 0.9,
            "metadata": {
                "doc_id": 10,
                "sub_query_ids": ["SQ1"],
            },
        },
        {
            "chunk_id": 2,
            "page_content": "待审核售后单需要上传商品照片。",
            "score": 0.8,
            "metadata": {
                "doc_id": 11,
                "sub_query_ids": ["SQ2"],
            },
        },
    ]

    result = builder.build(
        documents,
        sub_queries=[
            {
                "sub_query_id": "SQ1",
                "question": "订单如何修改地址？",
            },
            {
                "sub_query_id": "SQ2",
                "question": "售后单需要什么材料？",
            },
        ],
    )

    assert "## 子问题 SQ1：订单如何修改地址？" in result.context
    assert "## 子问题 SQ2：售后单需要什么材料？" in result.context
    assert result.context.index("[C1]") < result.context.index("[C2]")


def test_shared_chunk_uses_one_citation_across_sub_query_groups() -> None:
    builder = make_builder()
    result = builder.build(
        [
            {
                "chunk_id": 1,
                "page_content": "该规则同时说明地址修改和材料要求。",
                "score": 0.9,
                "metadata": {
                    "doc_id": 10,
                    "sub_query_ids": ["SQ1", "SQ2"],
                },
            }
        ],
        sub_queries=[
            {"sub_query_id": "SQ1", "question": "地址条件？"},
            {"sub_query_id": "SQ2", "question": "材料要求？"},
        ],
    )

    assert len(result.citations) == 1
    assert result.citations[0].citation_id == "C1"
    assert result.context.count("[C1]") == 2


def test_simple_context_keeps_flat_rendering() -> None:
    builder = make_builder()
    result = builder.build(
        [
            {
                "chunk_id": 1,
                "page_content": "退款规则正文。",
                "score": 0.9,
                "metadata": {"doc_id": 10},
            }
        ]
    )

    assert result.context == "[C1]\n退款规则正文。"
    assert "## 子问题" not in result.context
