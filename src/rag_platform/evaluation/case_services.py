import json
from typing import Any, Protocol

from pydantic import ValidationError

from src.rag_platform.core.exceptions import ModelResponseFormatError
from src.rag_platform.evaluation.case_models import (
    CaseQuotaPlan,
    CaseReviewBatch,
    CaseSeed,
    CaseSourceDocument,
    GeneratedCaseBatch,
    GeneratedCaseText,
    NoAnswerSubtype,
)
from src.rag_platform.evaluation.case_validation import (
    normalize_question,
    validate_generated_case,
)
from src.rag_platform.evaluation.models import (
    EvidenceSpec,
    EvalCaseStatus,
    EvalCaseType,
    GeneratedEvalCase,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
)


class JsonChatClient(Protocol):
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


class CaseGenerationService:
    def __init__(
        self,
        *,
        client: JsonChatClient,
        prompt_template: str,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.max_attempts = max_attempts

    async def generate_batch(
        self,
        *,
        seeds: list[CaseSeed],
        documents: dict[str, CaseSourceDocument],
    ) -> list[GeneratedEvalCase]:
        if not seeds or len(seeds) > 5:
            raise ValueError("每批必须包含1到5个评测题种子")

        source_codes = {
            code for seed in seeds for code in seed.source_doc_codes
        }
        source_payload = [
            documents[code].model_dump(mode="json")
            for code in sorted(source_codes)
        ]
        seed_payload = [
            seed.model_dump(mode="json")
            for seed in seeds
        ]
        base_prompt = self.prompt_template.format(
            seed_batch_json=_json_text(seed_payload),
            source_documents_json=_json_text(source_payload),
        )
        last_error: Exception | None = None
        validation_feedback: dict[str, list[str]] = {}
        for _ in range(self.max_attempts):
            prompt = base_prompt
            if validation_feedback:
                prompt += (
                    "\n\n上一轮输出未通过程序校验，请修正以下问题：\n"
                    + _json_text(validation_feedback)
                )
            try:
                payload = await self.client.chat_json(
                    system_prompt="你是RAG评测题生成器，只能依据给定事实出题。",
                    user_prompt=prompt,
                    temperature=0.4,
                    max_tokens=6144,
                )
                batch = GeneratedCaseBatch.model_validate(payload)
                generated_by_seed = {
                    item.seed_code: item for item in batch.cases
                }
                expected_codes = {seed.seed_code for seed in seeds}
                if set(generated_by_seed) != expected_codes:
                    raise ValueError(
                        "模型返回的seed_code集合与请求不一致"
                    )
                cases = [
                    self.build_case(
                        seed=seed,
                        generated=generated_by_seed[seed.seed_code],
                    )
                    for seed in seeds
                ]
                validation_feedback = {
                    case.case_code: errors
                    for case, seed in zip(cases, seeds)
                    if (
                        errors := validate_generated_case(
                            case=case,
                            seed=seed,
                            documents=documents,
                        )
                    )
                }
                if validation_feedback:
                    raise ValueError(
                        f"业务校验失败：{validation_feedback}"
                    )
                return cases
            except (
                ModelResponseFormatError,
                ValidationError,
                ValueError,
            ) as exc:
                last_error = exc
        raise ValueError(
            f"评测题批量生成连续失败：{last_error}"
        ) from last_error

    @staticmethod
    def build_case(
        *,
        seed: CaseSeed,
        generated: GeneratedCaseText | dict,
    ) -> GeneratedEvalCase:
        text = (
            generated
            if isinstance(generated, GeneratedCaseText)
            else GeneratedCaseText.model_validate(generated)
        )
        evidences = [
            EvidenceSpec(
                source_doc_code=fact.source_doc_code,
                evidence_quote=fact.fact_text,
                fact_key=fact.fact_key,
                relevance_grade=3,
                mapping_status=MappingStatus.PENDING,
            )
            for fact in seed.facts
        ]
        reference_answer = (
            text.reference_answer if evidences else None
        )
        return GeneratedEvalCase(
            case_code=seed.seed_code,
            question=text.question,
            normalized_question=normalize_question(text.question),
            reference_answer=reference_answer,
            case_type=seed.case_type,
            target_doc_types=seed.target_doc_types,
            expected_action=seed.expected_action,
            difficulty=text.difficulty,
            dataset_split=seed.dataset_split,
            business_domain="ecommerce_after_sales",
            required_fact_count=len({fact.fact_key for fact in seed.facts}),
            generation_metadata={
                "seed_code": seed.seed_code,
                "source_group": seed.source_group,
                "source_doc_codes": seed.source_doc_codes,
                "source_topics": seed.source_topics,
                "required_identifier": seed.required_identifier,
                "version_group": seed.version_group,
                "no_answer_subtype": (
                    seed.no_answer_subtype.value
                    if seed.no_answer_subtype
                    else None
                ),
            },
            evidences=evidences,
        )


class CaseReviewService:
    def __init__(
        self,
        *,
        reviewer: JsonChatClient,
        prompt_template: str,
        max_attempts: int = 2,
    ) -> None:
        self.reviewer = reviewer
        self.prompt_template = prompt_template
        self.max_attempts = max_attempts

    async def review_batch(
        self,
        *,
        cases: list[GeneratedEvalCase],
        documents: dict[str, CaseSourceDocument],
    ) -> list[ReviewedEvalCase]:
        if not cases or len(cases) > 5:
            raise ValueError("每批必须包含1到5道待审核评测题")
        source_codes = {
            code
            for case in cases
            for code in case.generation_metadata.get(
                "source_doc_codes",
                [],
            )
        }
        prompt = self.prompt_template.format(
            cases_json=_json_text(
                [case.model_dump(mode="json") for case in cases]
            ),
            source_documents_json=_json_text(
                [
                    documents[code].model_dump(mode="json")
                    for code in sorted(source_codes)
                ]
            ),
        )
        expected_codes = {case.case_code for case in cases}
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                payload = await self.reviewer.chat_json(
                    system_prompt="你是独立的RAG评测题质量审核员。",
                    user_prompt=prompt,
                    temperature=0,
                    max_tokens=6144,
                )
                batch = CaseReviewBatch.model_validate(payload)
                review_by_code = {
                    review.case_code: review for review in batch.reviews
                }
                if set(review_by_code) != expected_codes:
                    raise ValueError(
                        "Qwen返回的case_code集合与请求不一致"
                    )
                reviewed = []
                for case in cases:
                    review = review_by_code[case.case_code]
                    passed = review.passed
                    reviewed.append(
                        ReviewedEvalCase.model_validate(
                            {
                                **case.model_dump(mode="json"),
                                "difficulty": review.difficulty.value,
                                "review_status": (
                                    ReviewStatus.PASSED.value
                                    if passed
                                    else ReviewStatus.REJECTED.value
                                ),
                                "review_score": review.score,
                                "review_reason": (
                                    "审核通过"
                                    if passed
                                    else "；".join(review.issues)
                                    or "未满足评测题审核阈值"
                                ),
                                "status": (
                                    EvalCaseStatus.ACTIVE.value
                                    if passed
                                    else EvalCaseStatus.DISABLED.value
                                ),
                                "generation_metadata": {
                                    **case.generation_metadata,
                                    "qwen_review": review.model_dump(
                                        mode="json"
                                    ),
                                },
                            }
                        )
                    )
                return reviewed
            except (
                ModelResponseFormatError,
                ValidationError,
                ValueError,
            ) as exc:
                last_error = exc
        raise ValueError(f"评测题审核连续失败：{last_error}") from last_error


def select_reviewed_cases(
    reviewed_cases: list[ReviewedEvalCase],
    plan: CaseQuotaPlan,
) -> list[ReviewedEvalCase]:
    selected: list[ReviewedEvalCase] = []
    sequence_by_type: dict[EvalCaseType, int] = {}
    deficits: list[str] = []

    for split, type_counts in plan.split_case_type_counts.items():
        for case_type, required_count in type_counts.items():
            if case_type == EvalCaseType.NO_ANSWER:
                subtype_counts = plan.split_no_answer_subtype_counts[split]
                for subtype, subtype_count in subtype_counts.items():
                    candidates = _eligible_cases(
                        reviewed_cases,
                        split=split,
                        case_type=case_type,
                        subtype=subtype,
                    )
                    if len(candidates) < subtype_count:
                        deficits.append(
                            f"{split.value}/{case_type.value}/"
                            f"{subtype.value}:"
                            f"{len(candidates)}/{subtype_count}"
                        )
                        continue
                    selected.extend(candidates[:subtype_count])
            else:
                candidates = _eligible_cases(
                    reviewed_cases,
                    split=split,
                    case_type=case_type,
                )
                if len(candidates) < required_count:
                    deficits.append(
                        f"{split.value}/{case_type.value}:"
                        f"{len(candidates)}/{required_count}"
                    )
                    continue
                selected.extend(candidates[:required_count])

    if deficits:
        raise ValueError("审核通过题目配额不足：" + "；".join(deficits))

    renumbered = []
    for case in selected:
        sequence_by_type[case.case_type] = (
            sequence_by_type.get(case.case_type, 0) + 1
        )
        code = (
            f"CASE_{case.case_type.value}_"
            f"{sequence_by_type[case.case_type]:03d}"
        )
        renumbered.append(case.model_copy(update={"case_code": code}))
    return renumbered


def retain_reviews_for_recheck(
    reviewed_cases: list[ReviewedEvalCase],
    *,
    recheck_rejected: bool,
    case_type: EvalCaseType | None = None,
) -> list[ReviewedEvalCase]:
    if not recheck_rejected:
        return reviewed_cases
    return [
        case
        for case in reviewed_cases
        if not (
            case.review_status == ReviewStatus.REJECTED
            and (case_type is None or case.case_type == case_type)
        )
    ]


def _eligible_cases(
    reviewed_cases: list[ReviewedEvalCase],
    *,
    split,
    case_type,
    subtype: NoAnswerSubtype | None = None,
) -> list[ReviewedEvalCase]:
    candidates = [
        case
        for case in reviewed_cases
        if case.dataset_split == split
        and case.case_type == case_type
        and case.review_status == ReviewStatus.PASSED
        and case.status == EvalCaseStatus.ACTIVE
        and (
            subtype is None
            or case.generation_metadata.get("no_answer_subtype")
            == subtype.value
        )
    ]
    return sorted(
        candidates,
        key=lambda case: (
            -(case.review_score or 0.0),
            case.case_code,
        ),
    )


def _json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
