from src.rag_platform.evaluation.case_models import (
    CaseSeed,
    CaseSourceDocument,
    CaseSourceFact,
)
from src.rag_platform.evaluation.case_services import CaseGenerationService
from src.rag_platform.evaluation.case_validation import (
    SemanticDeduplicator,
    classify_grouped_semantic_duplicates,
    validate_generated_case,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    ExpectedAction,
    SourceDocumentType,
)


class FakeJsonClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class SequencedJsonClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.payloads.pop(0)


def _fact(
    *,
    source_doc_code: str = "FAQ_ORDER_STATUS_001",
    fact_key: str = "order_cancel_pending_payment",
    fact_text: str = "待支付订单可由用户直接取消。",
) -> CaseSourceFact:
    return CaseSourceFact(
        source_doc_code=source_doc_code,
        fact_key=fact_key,
        fact_text=fact_text,
        chunk_ids=[101],
    )


def _document() -> CaseSourceDocument:
    return CaseSourceDocument(
        source_doc_code="FAQ_ORDER_STATUS_001",
        mapped_doc_id=11,
        doc_type=SourceDocumentType.FAQ,
        title="订单状态与取消常见问题",
        topic="order",
        version="1.0",
        required_identifiers=["F-ORDER-001", "ORDER_CANCEL_PENDING"],
        facts=[_fact()],
    )


def _seed(case_type: EvalCaseType = EvalCaseType.DIRECT) -> CaseSeed:
    return CaseSeed(
        seed_code=f"SEED_{case_type.value}_001",
        case_type=case_type,
        dataset_split=DatasetSplit.DEVELOPMENT,
        expected_action=ExpectedAction.ANSWER,
        source_doc_codes=["FAQ_ORDER_STATUS_001"],
        source_topics=["order"],
        source_group="topic:order",
        facts=[_fact()],
        target_doc_types=[SourceDocumentType.FAQ],
        required_identifier=(
            "F-ORDER-001"
            if case_type == EvalCaseType.EXACT
            else None
        ),
        variant_index=1,
    )


async def test_generation_service_uses_canonical_catalog_evidence() -> None:
    client = FakeJsonClient(
        {
            "cases": [
                {
                    "seed_code": "SEED_DIRECT_001",
                    "question": "待支付订单应该怎样取消？",
                    "reference_answer": "用户可以直接取消待支付订单。",
                    "difficulty": "EASY",
                }
            ]
        }
    )
    service = CaseGenerationService(
        client=client,
        prompt_template="{seed_batch_json}\n{source_documents_json}",
    )

    cases = await service.generate_batch(
        seeds=[_seed()],
        documents={"FAQ_ORDER_STATUS_001": _document()},
    )

    assert len(cases) == 1
    assert cases[0].evidences[0].fact_key == "order_cancel_pending_payment"
    assert cases[0].evidences[0].evidence_quote == (
        "待支付订单可由用户直接取消。"
    )
    assert cases[0].evidences[0].source_doc_code == "FAQ_ORDER_STATUS_001"
    assert cases[0].expected_action == ExpectedAction.ANSWER
    assert cases[0].generation_metadata["source_group"] == "topic:order"


async def test_generation_service_rejects_missing_seed_output() -> None:
    client = FakeJsonClient(
        {
            "cases": [
                {
                    "seed_code": "UNKNOWN",
                    "question": "问题",
                    "reference_answer": "答案",
                    "difficulty": "EASY",
                }
            ]
        }
    )
    service = CaseGenerationService(client=client, prompt_template="{seed_batch_json}")

    try:
        await service.generate_batch(
            seeds=[_seed()],
            documents={"FAQ_ORDER_STATUS_001": _document()},
        )
    except ValueError as exc:
        assert "seed_code" in str(exc)
    else:
        raise AssertionError("缺失seed输出时必须失败")


async def test_generation_service_retries_with_business_validation_feedback() -> None:
    client = SequencedJsonClient(
        [
            {
                "cases": [
                    {
                        "seed_code": "SEED_DIRECT_001",
                        "question": "《订单状态与取消常见问题》里怎么取消？",
                        "reference_answer": "待支付订单可直接取消。",
                        "difficulty": "EASY",
                    }
                ]
            },
            {
                "cases": [
                    {
                        "seed_code": "SEED_DIRECT_001",
                        "question": "待支付订单应该怎样取消？",
                        "reference_answer": "待支付订单可直接取消。",
                        "difficulty": "EASY",
                    }
                ]
            },
        ]
    )
    service = CaseGenerationService(
        client=client,
        prompt_template="{seed_batch_json}\n{source_documents_json}",
    )

    cases = await service.generate_batch(
        seeds=[_seed()],
        documents={"FAQ_ORDER_STATUS_001": _document()},
    )

    assert cases[0].question == "待支付订单应该怎样取消？"
    assert len(client.calls) == 2
    assert "QUESTION_LEAKS_DOCUMENT_TITLE" in client.calls[1]["user_prompt"]


def test_exact_case_requires_identifier_in_question() -> None:
    case = CaseGenerationService.build_case(
        seed=_seed(EvalCaseType.EXACT),
        generated={
            "seed_code": "SEED_EXACT_001",
            "question": "这个订单编号代表什么？",
            "reference_answer": "待支付订单可直接取消。",
            "difficulty": "MEDIUM",
        },
    )

    errors = validate_generated_case(
        case=case,
        seed=_seed(EvalCaseType.EXACT),
        documents={"FAQ_ORDER_STATUS_001": _document()},
    )

    assert "EXACT_QUESTION_MISSING_IDENTIFIER" in errors


def test_question_must_not_expose_source_document_title() -> None:
    case = CaseGenerationService.build_case(
        seed=_seed(),
        generated={
            "seed_code": "SEED_DIRECT_001",
            "question": "《订单状态与取消常见问题》里待支付订单怎么取消？",
            "reference_answer": "待支付订单可直接取消。",
            "difficulty": "EASY",
        },
    )

    errors = validate_generated_case(
        case=case,
        seed=_seed(),
        documents={"FAQ_ORDER_STATUS_001": _document()},
    )

    assert "QUESTION_LEAKS_DOCUMENT_TITLE" in errors


def test_semantic_deduplicator_rejects_high_similarity_and_marks_borderline() -> None:
    result = SemanticDeduplicator(
        direct_threshold=0.92,
        review_threshold=0.85,
    ).classify(
        vectors=[
            [1.0, 0.0],
            [0.99, 0.01],
            [0.88, 0.47],
            [0.0, 1.0],
        ]
    )

    assert result[0].decision == "KEEP"
    assert result[1].decision == "DUPLICATE"
    assert result[2].decision == "REVIEW"
    assert result[3].decision == "KEEP"


def test_semantic_dedup_keeps_same_meaning_across_different_case_types() -> None:
    decisions = classify_grouped_semantic_duplicates(
        vectors=[
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ],
        group_keys=[
            ("DEVELOPMENT", "DIRECT"),
            ("DEVELOPMENT", "PARAPHRASE"),
            ("DEVELOPMENT", "DIRECT"),
        ],
    )

    assert decisions[0].decision == "KEEP"
    assert decisions[1].decision == "KEEP"
    assert decisions[2].decision == "DUPLICATE"


def test_checked_in_case_prompts_support_python_format() -> None:
    project_root = Path(__file__).resolve().parents[2]
    generate_prompt = (
        project_root / "evaluation/prompts/case_generate.txt"
    ).read_text(encoding="utf-8")
    review_prompt = (
        project_root / "evaluation/prompts/case_review.txt"
    ).read_text(encoding="utf-8")

    generate_prompt.format(
        seed_batch_json="[]",
        source_documents_json="[]",
    )
    review_prompt.format(
        cases_json="[]",
        source_documents_json="[]",
    )
from pathlib import Path
