from types import SimpleNamespace

from src.rag_platform.application.chat_service import ChatService
from src.rag_platform.domain.answer_action import (
    AnswerAction,
    AnswerDecisionSource,
)
from src.rag_platform.schemas.answer_action import AnswerActionDecision
from src.rag_platform.schemas.chat_v2 import ChatRequestV2
from src.rag_platform.schemas.rag_workflow import RagRetrievalWorkflowResponse


class FakeWorkflowService:
    def __init__(
        self,
        context: str,
        *,
        query_analysis: dict | None = None,
        decomposition: dict | None = None,
    ) -> None:
        self.context = context
        self.query_analysis = query_analysis or {}
        self.decomposition = decomposition or {}
        self.requests = []

    async def run_retrieval_workflow(self, request):
        self.requests.append(request)
        return RagRetrievalWorkflowResponse(
            trace_id="trace-1",
            question=request.question,
            query_analysis=self.query_analysis,
            rewritten_question="退款规则",
            status="CONTEXT_READY" if self.context else "RETRIEVAL_INSUFFICIENT",
            retrieval_quality={"quality": "GOOD" if self.context else "POOR"},
            decomposition=self.decomposition,
            sub_query_coverage={},
            context=self.context,
            citations=(
                [{"citation_id": "C1", "chunk_id": 101}]
                if self.context
                else []
            ),
            context_build_info={
                "context_log_id": 12,
                "estimated_tokens": 20,
            },
        )


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.generate_calls += 1
        self.calls.append(kwargs)
        return "可以按照退款规则处理。[C1]"

    async def stream_generate(self, **kwargs):
        self.generate_calls += 1
        self.calls.append(kwargs)
        yield "可以按照"
        yield "退款规则处理。[C1]"


class FakeActionDecisionService:
    def __init__(self, decision: AnswerActionDecision) -> None:
        self.decision = decision
        self.calls: list[dict] = []

    async def decide(self, **kwargs) -> AnswerActionDecision:
        self.calls.append(kwargs)
        return self.decision


def decision(
    action: AnswerAction,
    *,
    clarification_question: str | None = None,
) -> AnswerActionDecision:
    return AnswerActionDecision(
        action=action,
        confidence=0.95,
        reason="测试动作决策",
        clarification_question=clarification_question,
        missing_information=[],
        decision_source=AnswerDecisionSource.LLM_EVIDENCE,
    )


class FakeAnswerRepository:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.citations: list[dict] = []
        self.updated: list[dict] = []

    def create_answer_log(self, **kwargs) -> int:
        self.created.append(kwargs)
        return 99

    def save_answer_citations(self, **kwargs) -> None:
        self.citations.append(kwargs)

    def update_answer_log(self, **kwargs) -> None:
        self.updated.append(kwargs)


class FakeCitationValidator:
    def validate(self, **kwargs) -> dict:
        return {"passed": True, "answer": kwargs["answer"]}


def make_settings():
    return SimpleNamespace(
        answer_model="fake-model",
        answer_temperature=0.0,
        answer_max_tokens=128,
        answer_require_citation=True,
        answer_fail_when_context_empty=True,
        action_decision_enabled=False,
        action_decision_clarify_threshold=0.75,
        action_decision_refuse_threshold=0.8,
        action_decision_max_attempts=2,
        action_decision_model="deepseek-chat",
    )


async def test_chat_uses_injected_dependencies_and_saves_success_result() -> None:
    workflow = FakeWorkflowService(context="退款规则正文")
    generator = FakeAnswerGenerator()
    repository = FakeAnswerRepository()
    service = ChatService(
        settings=make_settings(),
        workflow_service=workflow,
        answer_generator=generator,
        answer_repository=repository,
        citation_validator=FakeCitationValidator(),
    )

    response = await service.chat(
        ChatRequestV2(
            question="怎么退款？",
            session_id="session-1",
            business_domain="after_sales",
            top_k=5,
        )
    )

    assert workflow.requests[0].top_k == 5
    assert generator.generate_calls == 1
    assert repository.created[0]["status"] == "STREAMING"
    assert repository.citations[0]["answer_log_id"] == 99
    assert repository.updated[0]["status"] == "SUCCESS"
    assert response.answer_log_id == 99
    assert response.status == "SUCCESS"
    assert response.citation_validation["passed"] is True
    assert response.action_decision["action"] == "ANSWER"


async def test_execute_returns_internal_workflow_observation() -> None:
    workflow = FakeWorkflowService(context="退款规则正文")
    service = ChatService(
        settings=make_settings(),
        workflow_service=workflow,
        answer_generator=FakeAnswerGenerator(),
        answer_repository=FakeAnswerRepository(),
        citation_validator=FakeCitationValidator(),
    )

    execution = await service.execute(
        ChatRequestV2(question="怎么退款？")
    )

    assert execution.response.status == "SUCCESS"
    assert execution.workflow.context == "退款规则正文"
    assert execution.workflow.citations[0]["chunk_id"] == 101
    assert execution.latency_ms >= 0


async def test_chat_passes_sub_query_plan_to_answer_generator() -> None:
    generator = FakeAnswerGenerator()
    workflow = FakeWorkflowService(
        context="分组后的规则正文",
        decomposition={
            "requires_decomposition": True,
            "sub_queries": [
                {"sub_query_id": "SQ1", "question": "地址条件？"},
                {"sub_query_id": "SQ2", "question": "材料要求？"},
            ],
        },
    )
    service = ChatService(
        settings=make_settings(),
        workflow_service=workflow,
        answer_generator=generator,
        answer_repository=FakeAnswerRepository(),
        citation_validator=FakeCitationValidator(),
        action_decision_service=FakeActionDecisionService(
            decision(AnswerAction.ANSWER)
        ),
    )

    await service.chat(
        ChatRequestV2(
            question="怎么改地址，同时要补什么材料？"
        )
    )

    assert generator.calls[0]["sub_queries"] == [
        {"sub_query_id": "SQ1", "question": "地址条件？"},
        {"sub_query_id": "SQ2", "question": "材料要求？"},
    ]


async def test_chat_refuses_without_context_and_does_not_call_generator() -> None:
    workflow = FakeWorkflowService(context="")
    generator = FakeAnswerGenerator()
    repository = FakeAnswerRepository()
    service = ChatService(
        settings=make_settings(),
        workflow_service=workflow,
        answer_generator=generator,
        answer_repository=repository,
        citation_validator=FakeCitationValidator(),
    )

    response = await service.chat(ChatRequestV2(question="知识库外的问题"))

    assert generator.generate_calls == 0
    assert repository.created[0]["status"] == "REFUSED"
    assert repository.updated[0]["status"] == "REFUSED"
    assert repository.citations == []
    assert response.answer_log_id == 99
    assert response.status == "REFUSED"


async def test_chat_refuses_nonempty_context_when_decision_requires_it() -> None:
    workflow = FakeWorkflowService(context="只有相似主题，没有具体规则")
    generator = FakeAnswerGenerator()
    repository = FakeAnswerRepository()
    action_service = FakeActionDecisionService(
        decision(AnswerAction.REFUSE)
    )
    service = ChatService(
        settings=make_settings(),
        workflow_service=workflow,
        answer_generator=generator,
        answer_repository=repository,
        citation_validator=FakeCitationValidator(),
        action_decision_service=action_service,
    )

    response = await service.chat(
        ChatRequestV2(question="退货运费险赔付多少？")
    )

    assert generator.generate_calls == 0
    assert repository.created[0]["status"] == "REFUSED"
    assert repository.created[0]["citation_count"] == 0
    assert repository.citations == []
    assert response.status == "REFUSED"
    assert response.citations == []
    assert response.action_decision["action"] == "REFUSE"
    assert action_service.calls[0]["context"] == (
        "只有相似主题，没有具体规则"
    )


async def test_chat_returns_clarification_without_generating_answer() -> None:
    workflow = FakeWorkflowService(context="退款流程按签收状态区分")
    generator = FakeAnswerGenerator()
    repository = FakeAnswerRepository()
    service = ChatService(
        settings=make_settings(),
        workflow_service=workflow,
        answer_generator=generator,
        answer_repository=repository,
        citation_validator=FakeCitationValidator(),
        action_decision_service=FakeActionDecisionService(
            decision(
                AnswerAction.CLARIFY,
                clarification_question="商品是否已经签收？",
            )
        ),
    )

    response = await service.chat(
        ChatRequestV2(question="商品坏了怎么退款？")
    )

    assert generator.generate_calls == 0
    assert repository.created[0]["status"] == "CLARIFIED"
    assert repository.updated[0]["status"] == "CLARIFIED"
    assert repository.citations == []
    assert response.status == "CLARIFIED"
    assert response.answer == "商品是否已经签收？"
    assert response.citations == []
    assert response.action_decision["action"] == "CLARIFY"


async def test_stream_chat_returns_clarification_and_done_status() -> None:
    generator = FakeAnswerGenerator()
    repository = FakeAnswerRepository()
    service = ChatService(
        settings=make_settings(),
        workflow_service=FakeWorkflowService(
            context="退款流程按签收状态区分"
        ),
        answer_generator=generator,
        answer_repository=repository,
        citation_validator=FakeCitationValidator(),
        action_decision_service=FakeActionDecisionService(
            decision(
                AnswerAction.CLARIFY,
                clarification_question="订单是否已经发货？",
            )
        ),
    )

    events = [
        event
        async for event in service.stream_chat(
            ChatRequestV2(question="怎么取消订单？")
        )
    ]

    assert generator.generate_calls == 0
    assert any("订单是否已经发货？" in event for event in events)
    assert any('"status": "CLARIFIED"' in event for event in events)
