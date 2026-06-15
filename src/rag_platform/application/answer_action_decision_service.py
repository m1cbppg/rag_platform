import json
from typing import Any

from src.rag_platform.core.config import get_settings
from src.rag_platform.application.clarification_policy_service import (
    ClarificationPolicyService,
)
from src.rag_platform.application.evidence_constraint_service import (
    EvidenceConstraintService,
)
from src.rag_platform.domain.answer_action import (
    AnswerAction,
    AnswerDecisionSource,
)
from src.rag_platform.infrastructure.deepseek import DeepSeekClient
from src.rag_platform.rag.answer.action_decision_prompt import (
    ANSWERABILITY_DECISION_SYSTEM_PROMPT,
    ANSWERABILITY_DECISION_USER_PROMPT_TEMPLATE,
    CLARIFICATION_DECISION_SYSTEM_PROMPT,
    CLARIFICATION_DECISION_USER_PROMPT_TEMPLATE,
)
from src.rag_platform.schemas.answer_action import (
    AnswerActionDecision,
    AnswerabilityAssessment,
    ClarificationAssessment,
)


class AnswerActionDecisionService:
    def __init__(
        self,
        *,
        settings=None,
        client=None,
        client_factory=None,
        clarification_policy_service=None,
        evidence_constraint_service=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client
        self.client_factory = client_factory or DeepSeekClient
        self.clarification_policy_service = (
            clarification_policy_service
            or ClarificationPolicyService()
        )
        self.evidence_constraint_service = (
            evidence_constraint_service
            or EvidenceConstraintService()
        )

    async def decide(
        self,
        *,
        question: str,
        query_analysis: dict[str, Any],
        retrieval_quality: dict[str, Any],
        context: str,
        citations: list[dict[str, Any]],
    ) -> AnswerActionDecision:
        clarification = str(
            query_analysis.get("clarification_question") or ""
        ).strip()
        query_confidence = float(
            query_analysis.get("confidence") or 0.0
        )
        if (
            query_analysis.get("need_clarification") is True
            and clarification
            and query_confidence
            >= self.settings.action_decision_clarify_threshold
        ):
            return AnswerActionDecision(
                action=AnswerAction.CLARIFY,
                confidence=query_confidence,
                reason="Query分析确认用户问题缺少必要条件。",
                clarification_question=clarification,
                missing_information=[],
                decision_source=AnswerDecisionSource.QUERY_ANALYSIS,
            )

        policy_match = self.clarification_policy_service.detect(
            question=question
        )
        if policy_match is not None:
            return AnswerActionDecision(
                action=AnswerAction.CLARIFY,
                confidence=policy_match.confidence,
                reason=policy_match.reason,
                clarification_question=(
                    policy_match.clarification_question
                ),
                missing_information=(
                    policy_match.missing_slot_labels
                ),
                decision_source=AnswerDecisionSource.POLICY_ENGINE,
            )

        normalized_context = context.strip()
        if not normalized_context:
            return AnswerActionDecision(
                action=AnswerAction.REFUSE,
                confidence=1.0,
                reason="检索后没有可用于回答的知识库Context。",
                clarification_question=None,
                missing_information=["知识库证据"],
                decision_source=AnswerDecisionSource.EMPTY_CONTEXT,
            )

        constraint_gap = self.evidence_constraint_service.find_gap(
            question=question,
            context=normalized_context,
        )
        if constraint_gap is not None:
            missing = constraint_gap.missing_constraints
            return AnswerActionDecision(
                action=AnswerAction.REFUSE,
                confidence=0.99,
                reason=(
                    "问题包含Context未覆盖的精确限定条件："
                    + "、".join(missing)
                ),
                clarification_question=None,
                missing_information=missing,
                decision_source=AnswerDecisionSource.CONSTRAINT_GUARD,
            )

        if not self.settings.action_decision_enabled:
            return self._fallback_answer("动作决策功能未启用。")

        payload = {
            "question": question,
            "query_analysis": query_analysis,
            "retrieval_quality": retrieval_quality,
            "context": normalized_context,
            "citations": citations,
        }
        decision_input_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        client = self.client or self.client_factory()
        try:
            clarification_assessment = await self._call_model(
                client=client,
                system_prompt=CLARIFICATION_DECISION_SYSTEM_PROMPT,
                user_prompt=CLARIFICATION_DECISION_USER_PROMPT_TEMPLATE.format(
                    decision_input_json=decision_input_json
                ),
                schema=ClarificationAssessment,
            )
            clarification_decision = self._map_clarification(
                clarification_assessment
            )
            if clarification_decision is not None:
                return clarification_decision

            answerability_assessment = await self._call_model(
                client=client,
                system_prompt=ANSWERABILITY_DECISION_SYSTEM_PROMPT,
                user_prompt=ANSWERABILITY_DECISION_USER_PROMPT_TEMPLATE.format(
                    decision_input_json=decision_input_json
                ),
                schema=AnswerabilityAssessment,
            )
            return self._map_answerability(answerability_assessment)
        except Exception as exc:
            reason = str(exc).strip() or "未知错误"
            return self._fallback_answer(
                f"动作决策模型失败，保持现有回答行为：{reason}"
            )

    async def _call_model(
        self,
        *,
        client,
        system_prompt: str,
        user_prompt: str,
        schema,
    ):
        last_error: Exception | None = None
        for _ in range(self.settings.action_decision_max_attempts):
            try:
                raw = await client.chat_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.settings.action_decision_model,
                    temperature=0,
                    max_tokens=512,
                )
                return schema.model_validate(raw)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("动作决策模型未执行")

    def _map_clarification(
        self,
        assessment: ClarificationAssessment,
    ) -> AnswerActionDecision | None:
        clarification = str(
            assessment.clarification_question or ""
        ).strip()
        if (
            assessment.needs_clarification
            and assessment.confidence
            >= self.settings.action_decision_clarify_threshold
            and clarification
        ):
            return AnswerActionDecision(
                action=AnswerAction.CLARIFY,
                confidence=assessment.confidence,
                reason=assessment.reason,
                clarification_question=clarification,
                missing_information=assessment.missing_conditions,
                decision_source=AnswerDecisionSource.LLM_EVIDENCE,
            )
        return None

    def _map_answerability(
        self,
        assessment: AnswerabilityAssessment,
    ) -> AnswerActionDecision:
        if assessment.answerable:
            return AnswerActionDecision(
                action=AnswerAction.ANSWER,
                confidence=assessment.confidence,
                reason=assessment.reason,
                clarification_question=None,
                missing_information=assessment.missing_information,
                decision_source=AnswerDecisionSource.LLM_EVIDENCE,
            )

        if (
            assessment.confidence
            >= self.settings.action_decision_refuse_threshold
        ):
            return AnswerActionDecision(
                action=AnswerAction.REFUSE,
                confidence=assessment.confidence,
                reason=assessment.reason,
                clarification_question=None,
                missing_information=assessment.missing_information,
                decision_source=AnswerDecisionSource.LLM_EVIDENCE,
            )

        return self._fallback_answer(
            "证据不足判断置信度低于阈值，保持回答行为。",
            confidence=assessment.confidence,
        )

    @staticmethod
    def _fallback_answer(
        reason: str,
        *,
        confidence: float = 0.0,
    ) -> AnswerActionDecision:
        return AnswerActionDecision(
            action=AnswerAction.ANSWER,
            confidence=confidence,
            reason=reason,
            clarification_question=None,
            missing_information=[],
            decision_source=AnswerDecisionSource.FALLBACK,
        )
