from src.rag_platform.evaluation.judge_service import AnswerJudgeService
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    EvidenceSpec,
    ExpectedAction,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
)


class SequencedJudgeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.payloads.pop(0)


def _case(
    *,
    expected_action: ExpectedAction = ExpectedAction.ANSWER,
    case_type: EvalCaseType = EvalCaseType.DIRECT,
) -> ReviewedEvalCase:
    if expected_action == ExpectedAction.ANSWER:
        return ReviewedEvalCase(
            case_code="CASE_DIRECT_001",
            question="待支付订单如何取消？",
            reference_answer="待支付订单可以直接取消。",
            case_type=case_type,
            expected_action=expected_action,
            dataset_split=DatasetSplit.DEVELOPMENT,
            required_fact_count=1,
            evidences=[
                EvidenceSpec(
                    source_doc_code="FAQ_ORDER_STATUS_001",
                    evidence_quote="待支付订单可直接取消。",
                    fact_key="cancel_pending",
                    relevance_grade=3,
                    mapped_doc_id=1,
                    mapped_chunk_id=101,
                    mapping_status=MappingStatus.MAPPED,
                )
            ],
            review_status=ReviewStatus.PASSED,
        )
    return ReviewedEvalCase(
        case_code="CASE_NO_ANSWER_001",
        question="平台支持火星配送吗？",
        case_type=EvalCaseType.NO_ANSWER,
        expected_action=expected_action,
        dataset_split=DatasetSplit.TEST,
        required_fact_count=0,
        evidences=[],
        review_status=ReviewStatus.PASSED,
    )


def _answer_payload(score: float = 0.9) -> dict:
    return {
        "faithfulness_score": score,
        "answer_relevance_score": score,
        "completeness_score": score,
        "citation_entailment_score": score,
        "conflict_handling_score": None,
        "refusal_correct": None,
        "clarification_correct": None,
        "passed": False,
        "reasons": {
            "unsupported_claims": [],
            "missing_facts": [],
            "citation_issues": [],
        },
    }


async def test_answer_judge_uses_programmatic_pass_threshold() -> None:
    client = SequencedJudgeClient([_answer_payload(0.9)])
    score = await AnswerJudgeService(
        client=client,
        prompt_template="{judge_input_json}",
    ).judge(
        case=_case(),
        system_answer="待支付订单可以直接取消。[C1]",
        actual_action="ANSWER",
        context_blocks=["[C1] 待支付订单可直接取消。"],
        citations=[{"citation_id": "C1", "chunk_id": 101}],
    )

    assert score.passed is True
    assert score.faithfulness_score == 0.9
    assert score.raw_response["attempts"][0]["passed"] is False


async def test_borderline_score_triggers_reversed_context_review() -> None:
    client = SequencedJudgeClient(
        [_answer_payload(0.78), _answer_payload(0.82)]
    )
    service = AnswerJudgeService(
        client=client,
        prompt_template="{judge_input_json}",
    )

    score = await service.judge(
        case=_case(),
        system_answer="答案",
        actual_action="ANSWER",
        context_blocks=["第一段", "第二段"],
        citations=[],
    )

    assert len(client.calls) == 2
    assert client.calls[0]["user_prompt"].find("第一段") < (
        client.calls[0]["user_prompt"].find("第二段")
    )
    assert client.calls[1]["user_prompt"].find("第二段") < (
        client.calls[1]["user_prompt"].find("第一段")
    )
    assert score.faithfulness_score == 0.8
    assert score.passed is True


async def test_refusal_case_uses_refusal_boolean() -> None:
    client = SequencedJudgeClient(
        [
            {
                "faithfulness_score": None,
                "answer_relevance_score": None,
                "completeness_score": None,
                "citation_entailment_score": None,
                "conflict_handling_score": None,
                "refusal_correct": True,
                "clarification_correct": None,
                "passed": False,
                "reasons": {},
            }
        ]
    )

    score = await AnswerJudgeService(
        client=client,
        prompt_template="{judge_input_json}",
    ).judge(
        case=_case(expected_action=ExpectedAction.REFUSE),
        system_answer="知识库没有相关信息。",
        actual_action="REFUSE",
        context_blocks=[],
        citations=[],
    )

    assert score.refusal_correct is True
    assert score.passed is True
    assert len(client.calls) == 1


async def test_judge_retries_invalid_payload_once() -> None:
    client = SequencedJudgeClient(
        [{"unexpected": "payload"}, _answer_payload(0.9)]
    )

    score = await AnswerJudgeService(
        client=client,
        prompt_template="{judge_input_json}",
    ).judge(
        case=_case(),
        system_answer="答案",
        actual_action="ANSWER",
        context_blocks=["证据"],
        citations=[],
    )

    assert score.passed is True
    assert len(client.calls) == 2
