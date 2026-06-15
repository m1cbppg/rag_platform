from types import SimpleNamespace

from src.rag_platform.rag.adaptive.query_decomposer import QueryDecomposer


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
        query_decomposition_enabled=True,
        query_decomposition_model="deepseek-chat",
        query_decomposition_max_sub_queries=3,
        query_decomposition_max_attempts=2,
        query_decomposition_min_query_length=18,
        query_decomposition_min_benefit_score=0.8,
        query_decomposition_allow_dependent=False,
    )


async def test_simple_question_skips_model_and_keeps_original_query() -> None:
    client = FakeClient()
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="退款多久到账？",
        rewritten_question="退款到账时间",
        target_doc_types=["FAQ"],
    )

    assert result.requires_decomposition is False
    assert result.fallback_used is False
    assert result.sub_queries == []
    assert client.calls == []


async def test_complex_question_returns_deduplicated_atomic_sub_queries() -> None:
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "PARALLEL",
            "benefit_score": 0.95,
            "reason": "包含地址修改和售后材料两个信息需求",
            "sub_queries": [
                {
                    "question": "未出库订单修改地址需要满足什么条件？",
                    "target_doc_types": ["FAQ", "MANUAL"],
                },
                {
                    "question": "待审核售后单需要上传什么材料？",
                    "target_doc_types": ["FAQ", "SOP"],
                },
                {
                    "question": "待审核售后单需要上传什么材料？",
                    "target_doc_types": ["SOP"],
                },
            ],
        }
    )
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question=(
            "未出库的订单修改地址需要满足什么条件？"
            "如果该订单同时有一个待审核售后单，需要上传什么材料？"
        ),
        rewritten_question="未出库订单修改地址和待审核售后材料",
        target_doc_types=["FAQ", "MANUAL", "SOP"],
    )

    assert result.requires_decomposition is True
    assert result.fallback_used is False
    assert [item.sub_query_id for item in result.sub_queries] == [
        "SQ1",
        "SQ2",
    ]
    assert [item.question for item in result.sub_queries] == [
        "未出库订单修改地址需要满足什么条件？",
        "待审核售后单需要上传什么材料？",
    ]
    assert result.sub_queries[0].target_doc_types == ["FAQ", "MANUAL"]
    assert len(client.calls) == 1


async def test_decomposition_limits_unique_sub_queries_to_configured_max() -> None:
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "PARALLEL",
            "benefit_score": 0.9,
            "reason": "包含四个信息需求",
            "sub_queries": [
                {"question": f"子问题{i}", "target_doc_types": []}
                for i in range(1, 5)
            ],
        }
    )
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="请分别说明条件一、条件二、条件三以及条件四？",
        rewritten_question=None,
        target_doc_types=[],
    )

    assert [item.question for item in result.sub_queries] == [
        "子问题1",
        "子问题2",
        "子问题3",
    ]


async def test_invalid_model_response_falls_back_without_decomposition() -> None:
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "PARALLEL",
            "benefit_score": 0.9,
            "reason": "复杂问题",
            "sub_queries": [
                {
                    "question": "只有一个子问题",
                    "target_doc_types": ["FAQ"],
                }
            ],
        }
    )
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="退款失败时什么时候重试，以及什么时候升级人工复核？",
        rewritten_question="退款失败重试与人工复核条件",
        target_doc_types=["SOP", "RULE"],
    )

    assert result.requires_decomposition is False
    assert result.fallback_used is True
    assert result.sub_queries == []
    assert len(client.calls) == 2


async def test_model_failure_falls_back_and_factory_client_is_closed() -> None:
    client = FakeClient(error=RuntimeError("model unavailable"))
    decomposer = QueryDecomposer(
        settings=_settings(),
        client_factory=lambda: client,
    )

    result = await decomposer.decompose(
        question="退款失败时什么时候重试，以及什么时候升级人工复核？",
        rewritten_question="退款失败重试与人工复核条件",
        target_doc_types=["SOP", "RULE"],
    )

    assert result.requires_decomposition is False
    assert result.fallback_used is True
    assert "RuntimeError" in result.reason
    assert len(client.calls) == 2
    assert client.closed is True


async def test_disabled_decomposition_never_calls_model() -> None:
    settings = _settings()
    settings.query_decomposition_enabled = False
    client = FakeClient()
    decomposer = QueryDecomposer(settings=settings, client=client)

    result = await decomposer.decompose(
        question="退款失败时什么时候重试，以及什么时候升级人工复核？",
        rewritten_question="退款失败重试与人工复核条件",
        target_doc_types=["SOP", "RULE"],
    )

    assert result.requires_decomposition is False
    assert result.reason == "查询分解未启用"
    assert client.calls == []


async def test_single_conditional_question_does_not_call_model() -> None:
    client = FakeClient()
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="如果修改地址涉及跨省或者有风险，需要什么流程？",
        rewritten_question=None,
        target_doc_types=["SOP"],
    )

    assert result.requires_decomposition is False
    assert client.calls == []


async def test_comparison_and_choice_questions_do_not_call_model() -> None:
    client = FakeClient()
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    comparison = await decomposer.decompose(
        question="根据新旧订单取消规则，已发货订单分别有什么区别？",
        rewritten_question=None,
        target_doc_types=["RULE"],
    )
    choice = await decomposer.decompose(
        question="支付处理中超过30分钟，应该先取消？还是先人工核查？",
        rewritten_question=None,
        target_doc_types=["RULE", "SOP"],
    )

    assert comparison.requires_decomposition is False
    assert choice.requires_decomposition is False
    assert client.calls == []


async def test_business_status_update_is_not_treated_as_version_comparison() -> None:
    client = FakeClient(
        {
            "requires_decomposition": False,
            "decomposition_type": "NONE",
            "benefit_score": 0.2,
            "reason": "无需分解",
            "sub_queries": [],
        }
    )
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question=(
            "物流轨迹六十小时没有更新，先判断延迟类型，"
            "然后根据类型说明由谁核查？"
        ),
        rewritten_question=None,
        target_doc_types=["FAQ"],
    )

    assert result.requires_decomposition is False
    assert len(client.calls) == 1


async def test_clarification_request_does_not_call_model() -> None:
    client = FakeClient()
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="退款和换货分别怎么办？",
        rewritten_question=None,
        target_doc_types=["FAQ"],
        need_clarification=True,
    )

    assert result.requires_decomposition is False
    assert client.calls == []


async def test_low_benefit_result_is_skipped_without_fallback() -> None:
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "PARALLEL",
            "benefit_score": 0.55,
            "reason": "两个问题来自同一规则，拆分收益较低",
            "sub_queries": [
                {"question": "信用卡退款多久到账？"},
                {"question": "余额退款多久到账？"},
            ],
        }
    )
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="信用卡退款多久到账？余额退款多久到账？",
        rewritten_question=None,
        target_doc_types=["FAQ"],
    )

    assert result.requires_decomposition is False
    assert result.fallback_used is False
    assert result.benefit_score == 0.55
    assert len(client.calls) == 1


async def test_dependent_result_is_skipped_when_not_enabled() -> None:
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "DEPENDENT",
            "benefit_score": 0.95,
            "reason": "第二步查询依赖第一步结果",
            "sub_queries": [
                {"question": "先查询订单当前状态"},
                {"question": "根据订单状态判断后续流程"},
            ],
        }
    )
    decomposer = QueryDecomposer(settings=_settings(), client=client)

    result = await decomposer.decompose(
        question="先查订单状态，然后根据结果应该走什么处理流程？",
        rewritten_question=None,
        target_doc_types=["SOP"],
    )

    assert result.requires_decomposition is False
    assert result.fallback_used is False
    assert result.decomposition_type == "DEPENDENT"


async def test_dependent_result_returns_two_hop_template_when_enabled() -> None:
    settings = _settings()
    settings.query_decomposition_allow_dependent = True
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "DEPENDENT",
            "benefit_score": 0.95,
            "reason": "第二跳依赖错误码含义",
            "sub_queries": [
                {
                    "question": "F-REFUND-1003表示什么失败类型？",
                    "target_doc_types": ["RULE"],
                    "depends_on_sub_query_id": None,
                },
                {
                    "question": (
                        "{{intermediate_fact}}应该走什么处理流程？"
                    ),
                    "target_doc_types": ["SOP"],
                    "depends_on_sub_query_id": "SQ1",
                },
            ],
        }
    )
    decomposer = QueryDecomposer(settings=settings, client=client)

    result = await decomposer.decompose(
        question=(
            "先查询F-REFUND-1003表示什么失败类型，"
            "然后根据结果查询对应处理流程？"
        ),
        rewritten_question=None,
        target_doc_types=["RULE", "SOP"],
    )

    assert result.requires_decomposition is True
    assert result.decomposition_type == "DEPENDENT"
    assert result.sub_queries[1].depends_on_sub_query_id == "SQ1"
    assert result.sub_queries[1].is_template is True


async def test_dependent_aliases_and_field_shapes_are_normalized() -> None:
    settings = _settings()
    settings.query_decomposition_allow_dependent = True
    client = FakeClient(
        {
            "requires_decomposition": True,
            "decomposition_type": "sequential",
            "benefit_score": 0.9,
            "reason": "第二跳依赖第一跳",
            "sub_queries": [
                {
                    "question": "订单状态对应什么取消方式？",
                    "target_doc_types": "FAQ",
                    "depends_on_sub_query_id": "",
                },
                {
                    "question": (
                        "{{intermediate_fact}}如何在后台提交？"
                    ),
                    "target_doc_types": "MANUAL",
                    "depends_on_sub_query_id": "1",
                },
            ],
        }
    )
    decomposer = QueryDecomposer(settings=settings, client=client)

    result = await decomposer.decompose(
        question=(
            "先判断订单状态对应的取消方式，"
            "然后根据结果说明后台如何提交？"
        ),
        rewritten_question=None,
        target_doc_types=["FAQ", "MANUAL"],
    )

    assert result.requires_decomposition is True
    assert result.decomposition_type == "DEPENDENT"
    assert result.sub_queries[0].target_doc_types == ["FAQ"]
    assert result.sub_queries[1].depends_on_sub_query_id == "SQ1"
