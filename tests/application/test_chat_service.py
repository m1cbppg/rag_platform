from types import SimpleNamespace

from src.rag_platform.application.chat_service import ChatService
from src.rag_platform.schemas.chat_v2 import ChatRequestV2
from src.rag_platform.schemas.rag_workflow import RagRetrievalWorkflowResponse


class FakeWorkflowService:
    def __init__(self, context: str) -> None:
        self.context = context
        self.requests = []

    async def run_retrieval_workflow(self, request):
        self.requests.append(request)
        return RagRetrievalWorkflowResponse(
            trace_id="trace-1",
            question=request.question,
            rewritten_question="退款规则",
            status="CONTEXT_READY" if self.context else "RETRIEVAL_INSUFFICIENT",
            retrieval_quality={"quality": "GOOD" if self.context else "POOR"},
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

    async def generate(self, **kwargs) -> str:
        self.generate_calls += 1
        return "可以按照退款规则处理。[C1]"


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
