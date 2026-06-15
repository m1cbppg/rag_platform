from types import SimpleNamespace

from src.rag_platform.application.answer_action_decision_service import (
    AnswerActionDecisionService,
)
from src.rag_platform.domain.answer_action import (
    AnswerAction,
    AnswerDecisionSource,
)
from src.rag_platform.rag.answer.action_decision_prompt import (
    ANSWERABILITY_DECISION_SYSTEM_PROMPT,
)


class FakeDecisionClient:
    def __init__(
        self,
        response: dict | None = None,
        responses: list[dict] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return self.response or {}


class NoMatchPolicyService:
    def detect(self, *, question: str):
        return None


class MatchingPolicyService:
    def detect(self, *, question: str):
        return SimpleNamespace(
            policy_code="cancel_order",
            confidence=0.99,
            reason="取消订单必须先确认订单状态。",
            clarification_question="请问订单当前是什么状态？",
            missing_slot_labels=["订单状态"],
        )


def make_settings():
    return SimpleNamespace(
        action_decision_enabled=True,
        action_decision_model="deepseek-chat",
        action_decision_clarify_threshold=0.75,
        action_decision_refuse_threshold=0.80,
        action_decision_max_attempts=2,
    )


def test_answerability_prompt_rejects_unsupported_special_conditions() -> None:
    assert "不能用一般规则回答包含特殊限定条件的问题" in (
        ANSWERABILITY_DECISION_SYSTEM_PROMPT
    )
    assert "时间、金额、状态、版本、商品类型" in (
        ANSWERABILITY_DECISION_SYSTEM_PROMPT
    )
    assert "可以由多个 Context 证据块联合覆盖" in (
        ANSWERABILITY_DECISION_SYSTEM_PROMPT
    )
    assert "明确的特殊对象例外规则优先于一般规则" in (
        ANSWERABILITY_DECISION_SYSTEM_PROMPT
    )
    assert "不能用相邻业务流程代替用户明确询问的流程" in (
        ANSWERABILITY_DECISION_SYSTEM_PROMPT
    )
    assert "只提供相关概述但缺少所问的具体标准" in (
        ANSWERABILITY_DECISION_SYSTEM_PROMPT
    )


async def test_high_confidence_query_signal_returns_clarify_without_llm() -> None:
    client = FakeDecisionClient()
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="我的商品坏了，怎么办？",
        query_analysis={
            "need_clarification": True,
            "clarification_question": "商品是否已经签收？",
            "confidence": 0.9,
        },
        retrieval_quality={"quality": "GOOD"},
        context="售后规则正文",
        citations=[],
    )

    assert result.action == AnswerAction.CLARIFY
    assert result.decision_source == AnswerDecisionSource.QUERY_ANALYSIS
    assert result.clarification_question == "商品是否已经签收？"
    assert client.calls == []


async def test_empty_context_returns_refuse_without_llm() -> None:
    client = FakeDecisionClient()
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="股票代码是什么？",
        query_analysis={},
        retrieval_quality={"quality": "POOR"},
        context="  ",
        citations=[],
    )

    assert result.action == AnswerAction.REFUSE
    assert result.decision_source == AnswerDecisionSource.EMPTY_CONTEXT
    assert client.calls == []


async def test_llm_can_refuse_when_context_does_not_support_question() -> None:
    client = FakeDecisionClient(
        responses=[
            {
                "needs_clarification": False,
                "confidence": 0.9,
                "reason": "问题意图明确，不需要补充条件。",
                "clarification_question": None,
                "missing_conditions": [],
            },
            {
                "answerable": False,
                "confidence": 0.92,
                "reason": "Context没有运费险赔付标准。",
                "missing_information": ["赔付金额", "赔付条件"],
            },
        ]
    )
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="退货运费险怎么赔？",
        query_analysis={"confidence": 0.9},
        retrieval_quality={"quality": "GOOD"},
        context="这里只说明普通退货运费由谁承担。",
        citations=[{"citation_id": "C1", "chunk_id": 1}],
    )

    assert result.action == AnswerAction.REFUSE
    assert result.decision_source == AnswerDecisionSource.LLM_EVIDENCE
    assert result.missing_information == ["赔付金额", "赔付条件"]
    assert len(client.calls) == 2
    assert client.calls[0]["system_prompt"] != client.calls[1]["system_prompt"]


async def test_llm_can_request_clarification_with_question() -> None:
    client = FakeDecisionClient(
        response={
            "needs_clarification": True,
            "confidence": 0.88,
            "reason": "不同签收状态适用不同退款流程。",
            "clarification_question": "商品是否已经签收？",
            "missing_conditions": ["签收状态"],
        }
    )
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="商品坏了怎么退款？",
        query_analysis={"confidence": 0.7},
        retrieval_quality={"quality": "GOOD"},
        context="规则区分未签收和已签收两种处理方式。",
        citations=[],
    )

    assert result.action == AnswerAction.CLARIFY
    assert result.clarification_question == "商品是否已经签收？"
    assert result.decision_source == AnswerDecisionSource.LLM_EVIDENCE
    assert result.missing_information == ["签收状态"]
    assert len(client.calls) == 1


async def test_low_confidence_refuse_falls_back_to_answer() -> None:
    client = FakeDecisionClient(
        responses=[
            {
                "needs_clarification": False,
                "confidence": 0.9,
                "reason": "问题意图明确。",
                "clarification_question": None,
                "missing_conditions": [],
            },
            {
                "answerable": False,
                "confidence": 0.6,
                "reason": "不确定证据是否足够。",
                "missing_information": [],
            },
        ]
    )
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="怎么退款？",
        query_analysis={},
        retrieval_quality={"quality": "GOOD"},
        context="退款规则正文。",
        citations=[],
    )

    assert result.action == AnswerAction.ANSWER
    assert result.decision_source == AnswerDecisionSource.FALLBACK


async def test_llm_error_falls_back_to_answer_when_context_exists() -> None:
    client = FakeDecisionClient(error=RuntimeError("模型不可用"))
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="怎么退款？",
        query_analysis={},
        retrieval_quality={"quality": "GOOD"},
        context="退款规则正文。",
        citations=[],
    )

    assert result.action == AnswerAction.ANSWER
    assert result.decision_source == AnswerDecisionSource.FALLBACK
    assert "模型不可用" in result.reason
    assert len(client.calls) == 2


async def test_policy_match_returns_clarify_without_llm() -> None:
    client = FakeDecisionClient()
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=MatchingPolicyService(),
    )

    result = await service.decide(
        question="我的订单怎么取消？",
        query_analysis={},
        retrieval_quality={"quality": "GOOD"},
        context="不同订单状态对应不同取消流程。",
        citations=[],
    )

    assert result.action == AnswerAction.CLARIFY
    assert result.decision_source == AnswerDecisionSource.POLICY_ENGINE
    assert result.clarification_question == "请问订单当前是什么状态？"
    assert result.missing_information == ["订单状态"]
    assert client.calls == []


async def test_missing_exact_constraint_returns_refuse_without_llm() -> None:
    client = FakeDecisionClient()
    service = AnswerActionDecisionService(
        settings=make_settings(),
        client=client,
        clarification_policy_service=NoMatchPolicyService(),
    )

    result = await service.decide(
        question="下单后十分钟内取消，退款多久能到账？",
        query_analysis={},
        retrieval_quality={"quality": "GOOD"},
        context="已支付订单审核通过后1-3个工作日到账。",
        citations=[],
    )

    assert result.action == AnswerAction.REFUSE
    assert result.decision_source == (
        AnswerDecisionSource.CONSTRAINT_GUARD
    )
    assert result.missing_information == ["10分钟"]
    assert client.calls == []
