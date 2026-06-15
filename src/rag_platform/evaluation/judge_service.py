import json
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.rag_platform.core.exceptions import ModelResponseFormatError
from src.rag_platform.evaluation.models import (
    ActualAction,
    EvalCaseType,
    ExpectedAction,
    JudgeScore,
    ReviewedEvalCase,
)

ANSWER_JUDGE_PROMPT_VERSION = "v2-action-contract"


class JsonJudgeClient(Protocol):
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness_score: float | None = Field(ge=0, le=1)
    answer_relevance_score: float | None = Field(ge=0, le=1)
    completeness_score: float | None = Field(ge=0, le=1)
    citation_entailment_score: float | None = Field(ge=0, le=1)
    conflict_handling_score: float | None = Field(ge=0, le=1)
    refusal_correct: bool | None
    clarification_correct: bool | None
    passed: bool
    reasons: dict[str, Any]


class AnswerJudgeService:
    def __init__(
        self,
        *,
        client: JsonJudgeClient,
        prompt_template: str,
        provider: str = "dashscope",
        model: str = "qwen-plus",
        prompt_version: str = ANSWER_JUDGE_PROMPT_VERSION,
        pass_threshold: float = 0.8,
        borderline_margin: float = 0.05,
        max_attempts: int = 2,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.provider = provider
        self.model = model
        self.prompt_version = prompt_version
        self.pass_threshold = pass_threshold
        self.borderline_margin = borderline_margin
        self.max_attempts = max_attempts

    async def judge(
        self,
        *,
        case: ReviewedEvalCase,
        system_answer: str,
        actual_action: ActualAction | str,
        context_blocks: list[str],
        citations: list[dict[str, Any]],
    ) -> JudgeScore:
        start_time = time.perf_counter()
        first = await self._review_once(
            case=case,
            system_answer=system_answer,
            actual_action=ActualAction(actual_action),
            context_blocks=context_blocks,
            citations=citations,
        )
        attempts = [first.model_dump(mode="json")]
        final = first
        if self._is_borderline(case, first):
            second = await self._review_once(
                case=case,
                system_answer=system_answer,
                actual_action=ActualAction(actual_action),
                context_blocks=list(reversed(context_blocks)),
                citations=list(reversed(citations)),
            )
            attempts.append(second.model_dump(mode="json"))
            final = _average_responses(first, second)

        return JudgeScore(
            judge_provider=self.provider,
            judge_model=self.model,
            judge_prompt_version=self.prompt_version,
            faithfulness_score=final.faithfulness_score,
            answer_relevance_score=final.answer_relevance_score,
            completeness_score=final.completeness_score,
            citation_entailment_score=final.citation_entailment_score,
            conflict_handling_score=final.conflict_handling_score,
            refusal_correct=final.refusal_correct,
            clarification_correct=final.clarification_correct,
            passed=self._passed(case, final),
            reason=final.reasons,
            raw_response={"attempts": attempts},
            latency_ms=int((time.perf_counter() - start_time) * 1000),
        )

    async def _review_once(
        self,
        *,
        case: ReviewedEvalCase,
        system_answer: str,
        actual_action: ActualAction,
        context_blocks: list[str],
        citations: list[dict[str, Any]],
    ) -> JudgeResponse:
        payload = {
            "case_code": case.case_code,
            "case_type": case.case_type.value,
            "question": case.question,
            "expected_action": case.expected_action.value,
            "actual_action": actual_action.value,
            "reference_answer": case.reference_answer,
            "generation_metadata": case.generation_metadata,
            "system_answer": system_answer,
            "context_blocks": context_blocks,
            "citations": citations,
        }
        prompt = self.prompt_template.format(
            judge_input_json=json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                response = await self.client.chat_json(
                    system_prompt=(
                        "你是独立的RAG答案质量评审员，"
                        "只能依据给定评测输入评分。"
                    ),
                    user_prompt=prompt,
                    model=self.model,
                    temperature=0,
                    max_tokens=4096,
                )
                return JudgeResponse.model_validate(response)
            except (
                ModelResponseFormatError,
                ValidationError,
                ValueError,
            ) as exc:
                last_error = exc
        raise ValueError(f"答案评审连续失败：{last_error}") from last_error

    def _is_borderline(
        self,
        case: ReviewedEvalCase,
        response: JudgeResponse,
    ) -> bool:
        return any(
            abs(score - self.pass_threshold) <= self.borderline_margin
            for score in self._applicable_scores(case, response)
        )

    def _passed(
        self,
        case: ReviewedEvalCase,
        response: JudgeResponse,
    ) -> bool:
        if case.expected_action == ExpectedAction.REFUSE:
            return response.refusal_correct is True
        if case.expected_action == ExpectedAction.CLARIFY:
            return response.clarification_correct is True
        scores = self._applicable_scores(case, response)
        required_count = (
            5 if case.case_type == EvalCaseType.CONFLICT else 4
        )
        return (
            len(scores) == required_count
            and all(score >= self.pass_threshold for score in scores)
        )

    @staticmethod
    def _applicable_scores(
        case: ReviewedEvalCase,
        response: JudgeResponse,
    ) -> list[float]:
        if case.expected_action != ExpectedAction.ANSWER:
            return []
        values = [
            response.faithfulness_score,
            response.answer_relevance_score,
            response.completeness_score,
            response.citation_entailment_score,
        ]
        if case.case_type == EvalCaseType.CONFLICT:
            values.append(response.conflict_handling_score)
        return [value for value in values if value is not None]


def _average_responses(
    first: JudgeResponse,
    second: JudgeResponse,
) -> JudgeResponse:
    fields = (
        "faithfulness_score",
        "answer_relevance_score",
        "completeness_score",
        "citation_entailment_score",
        "conflict_handling_score",
    )
    update = {}
    for field in fields:
        left = getattr(first, field)
        right = getattr(second, field)
        if left is None or right is None:
            update[field] = left if right is None else right
        else:
            update[field] = round((left + right) / 2, 4)
    update.update(
        {
            "refusal_correct": (
                first.refusal_correct
                if first.refusal_correct == second.refusal_correct
                else False
            ),
            "clarification_correct": (
                first.clarification_correct
                if first.clarification_correct
                == second.clarification_correct
                else False
            ),
            "passed": first.passed and second.passed,
            "reasons": {
                "first": first.reasons,
                "reversed": second.reasons,
            },
        }
    )
    return first.model_copy(update=update)
