from types import SimpleNamespace

from src.rag_platform.rag.adaptive.intermediate_fact_extractor import (
    IntermediateFactExtractor,
)


class FakeClient:
    def __init__(
        self,
        response: dict | None = None,
        *,
        responses: list[dict] | None = None,
    ) -> None:
        self.responses = responses or [response or {}]
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        index = min(
            len(self.calls) - 1,
            len(self.responses) - 1,
        )
        return self.responses[index]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        dependent_fact_model="deepseek-chat",
        dependent_fact_min_confidence=0.75,
        dependent_fact_max_candidates=5,
        dependent_fact_max_attempts=1,
    )


async def test_extracts_evidence_bound_intermediate_fact() -> None:
    client = FakeClient(
        {
            "success": True,
            "intermediate_fact": "可重试退款失败",
            "evidence_quote": "F-REFUND-1003表示可重试退款失败",
            "supporting_chunk_id": 101,
            "confidence": 0.92,
            "reason": "错误码定义给出了失败类型",
        }
    )
    extractor = IntermediateFactExtractor(
        settings=_settings(),
        client=client,
    )

    result = await extractor.extract(
        question="F-REFUND-1003是什么失败，后续怎么处理？",
        first_hop_question="F-REFUND-1003表示什么？",
        next_query_template=(
            "{{intermediate_fact}}应该走什么处理流程？"
        ),
        candidate_documents=[
            {
                "chunk_id": 101,
                "title": "退款错误码",
                "page_content": (
                    "F-REFUND-1003表示可重试退款失败，"
                    "需要进入自动重试判断。"
                ),
            }
        ],
    )

    assert result.success is True
    assert result.intermediate_fact == "可重试退款失败"
    assert result.supporting_chunk_id == 101
    assert result.confidence == 0.92


async def test_rejects_quote_not_found_in_supporting_chunk() -> None:
    client = FakeClient(
        {
            "success": True,
            "intermediate_fact": "高风险订单",
            "evidence_quote": "该订单属于高风险订单",
            "supporting_chunk_id": 101,
            "confidence": 0.95,
            "reason": "风险规则",
        }
    )
    extractor = IntermediateFactExtractor(
        settings=_settings(),
        client=client,
    )

    result = await extractor.extract(
        question="该订单是什么风险等级，谁审批？",
        first_hop_question="该订单是什么风险等级？",
        next_query_template="{{intermediate_fact}}由谁审批？",
        candidate_documents=[
            {
                "chunk_id": 101,
                "page_content": "当前证据没有给出风险等级。",
            }
        ],
    )

    assert result.success is False
    assert result.fallback_used is True


async def test_connector_phrase_is_retried_with_validation_feedback() -> None:
    client = FakeClient(
        responses=[
            {
                "success": True,
                "intermediate_fact": "核实重复支付后",
                "evidence_quote": (
                    "核实重复支付后，多支付的款项原路退回。"
                ),
                "supporting_chunk_id": 65,
                "confidence": 0.95,
                "reason": "用于下一跳",
            },
            {
                "success": True,
                "intermediate_fact": (
                    "多余流水未绑定其他订单且退款账户"
                    "与支付账户一致"
                ),
                "evidence_quote": (
                    "确认多余流水未绑定其他订单，"
                    "且退款账户与支付账户一致。"
                ),
                "supporting_chunk_id": 179,
                "confidence": 0.95,
                "reason": "直接回答第一跳核实条件",
            },
        ]
    )
    settings = _settings()
    settings.dependent_fact_max_attempts = 2
    extractor = IntermediateFactExtractor(
        settings=settings,
        client=client,
    )

    result = await extractor.extract(
        question="先核实重复支付，再说明退款方式",
        first_hop_question="判定重复支付需要核实什么？",
        next_query_template="{{intermediate_fact}}后如何退款？",
        candidate_documents=[
            {
                "chunk_id": 65,
                "content": (
                    "核实重复支付后，多支付的款项原路退回。"
                ),
            },
            {
                "chunk_id": 179,
                "content": (
                    "确认多余流水未绑定其他订单，"
                    "且退款账户与支付账户一致。"
                ),
            },
        ],
    )

    assert result.success is True
    assert result.supporting_chunk_id == 179
    assert len(client.calls) == 2
    assert "流程连接短语" in client.calls[1]["user_prompt"]
