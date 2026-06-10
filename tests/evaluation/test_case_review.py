from src.rag_platform.evaluation.case_models import (
    CaseQuotaPlan,
    CaseReviewResult,
)
from src.rag_platform.evaluation.case_services import (
    CaseReviewService,
    retain_reviews_for_recheck,
    select_reviewed_cases,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    Difficulty,
    EvalCaseType,
    ExpectedAction,
    GeneratedEvalCase,
    ReviewStatus,
    ReviewedEvalCase,
)


class FakeReviewer:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def chat_json(self, **kwargs):
        return self.payload


def _case(
    *,
    code: str,
    case_type: EvalCaseType = EvalCaseType.NO_ANSWER,
    split: DatasetSplit = DatasetSplit.DEVELOPMENT,
    subtype: str = "KNOWLEDGE_GAP",
) -> GeneratedEvalCase:
    return GeneratedEvalCase(
        case_code=code,
        question=f"{code}对应的问题是什么？",
        case_type=case_type,
        expected_action=ExpectedAction.REFUSE,
        difficulty=Difficulty.MEDIUM,
        dataset_split=split,
        required_fact_count=0,
        generation_metadata={
            "source_group": f"group:{code}",
            "no_answer_subtype": subtype,
        },
        evidences=[],
    )


async def test_qwen_review_service_applies_pass_contract() -> None:
    case = _case(code="SEED_NO_ANSWER_001")
    reviewer = FakeReviewer(
        {
            "reviews": [
                {
                    "case_code": case.case_code,
                    "answerable": False,
                    "expected_action_correct": True,
                    "reference_answer_supported": True,
                    "evidence_complete": True,
                    "ambiguity_score": 0.05,
                    "difficulty": "MEDIUM",
                    "semantic_duplicate": False,
                    "issues": [],
                }
            ]
        }
    )

    reviewed = await CaseReviewService(
        reviewer=reviewer,
        prompt_template="{cases_json}\n{source_documents_json}",
    ).review_batch(cases=[case], documents={})

    assert reviewed[0].review_status == ReviewStatus.PASSED
    assert reviewed[0].review_score == 0.99
    assert reviewed[0].review_reason == "审核通过"


def test_review_result_rejects_high_ambiguity() -> None:
    review = CaseReviewResult(
        case_code="CASE_001",
        answerable=True,
        expected_action_correct=True,
        reference_answer_supported=True,
        evidence_complete=True,
        ambiguity_score=0.16,
        difficulty=Difficulty.MEDIUM,
        semantic_duplicate=False,
        issues=[],
    )

    assert review.passed is False


def test_recheck_retains_passed_and_other_case_types() -> None:
    cases = []
    for code, case_type, status in [
        ("PASSED_CONFLICT", EvalCaseType.CONFLICT, ReviewStatus.PASSED),
        ("REJECTED_CONFLICT", EvalCaseType.CONFLICT, ReviewStatus.REJECTED),
        ("REJECTED_DIRECT", EvalCaseType.DIRECT, ReviewStatus.REJECTED),
    ]:
        generated = _case(code=code, case_type=case_type)
        cases.append(
            ReviewedEvalCase.model_validate(
                {
                    **generated.model_dump(mode="json"),
                    "review_status": status,
                    "review_score": 0.8,
                    "review_reason": status.value,
                }
            )
        )

    retained = retain_reviews_for_recheck(
        cases,
        recheck_rejected=True,
        case_type=EvalCaseType.CONFLICT,
    )

    assert {case.case_code for case in retained} == {
        "PASSED_CONFLICT",
        "REJECTED_DIRECT",
    }


def test_selection_enforces_subtype_quota_and_reassigns_case_codes() -> None:
    plan = CaseQuotaPlan.model_validate(
        {
            "case_type_counts": {"NO_ANSWER": 300},
            "split_case_type_counts": {
                "DEVELOPMENT": {"NO_ANSWER": 180},
                "VALIDATION": {"NO_ANSWER": 60},
                "TEST": {"NO_ANSWER": 60},
            },
            "no_answer_subtype_counts": {
                "KNOWLEDGE_GAP": 150,
                "MISSING_CONDITION": 75,
                "OUT_OF_DOMAIN": 75,
            },
            "split_no_answer_subtype_counts": {
                "DEVELOPMENT": {
                    "KNOWLEDGE_GAP": 90,
                    "MISSING_CONDITION": 45,
                    "OUT_OF_DOMAIN": 45,
                },
                "VALIDATION": {
                    "KNOWLEDGE_GAP": 30,
                    "MISSING_CONDITION": 15,
                    "OUT_OF_DOMAIN": 15,
                },
                "TEST": {
                    "KNOWLEDGE_GAP": 30,
                    "MISSING_CONDITION": 15,
                    "OUT_OF_DOMAIN": 15,
                },
            },
            "split_topics": {
                "DEVELOPMENT": ["dev"],
                "VALIDATION": ["val"],
                "TEST": ["test"],
            },
        }
    )
    reviewed = []
    for split, counts in plan.split_no_answer_subtype_counts.items():
        for subtype, count in counts.items():
            for index in range(count):
                generated = _case(
                    code=f"SEED_{split.value}_{subtype.value}_{index}",
                    split=split,
                    subtype=subtype.value,
                )
                reviewed.append(
                    ReviewedEvalCase.model_validate(
                        {
                            **generated.model_dump(mode="json"),
                            "review_status": ReviewStatus.PASSED,
                            "review_score": 0.95,
                            "review_reason": "审核通过",
                        }
                    )
                )

    selected = select_reviewed_cases(reviewed, plan)

    assert len(selected) == 300
    assert selected[0].case_code == "CASE_NO_ANSWER_001"
    assert selected[-1].case_code == "CASE_NO_ANSWER_300"
