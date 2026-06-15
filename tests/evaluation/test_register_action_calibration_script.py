from types import SimpleNamespace

from scripts.register_action_calibration import (
    register_action_calibration,
)
from src.rag_platform.evaluation.case_persistence import (
    write_case_jsonl,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    ExpectedAction,
    ReviewStatus,
    ReviewedEvalCase,
)


class FakeDatasetRepository:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.cases: list[tuple[int, ReviewedEvalCase]] = []
        self.frozen: list[tuple[int, str]] = []

    def find_dataset(self, dataset_code, version):
        return None

    def create_dataset(self, **kwargs):
        self.created.append(kwargs)
        return 12

    def upsert_eval_case(self, dataset_id, case):
        self.cases.append((dataset_id, case))
        return len(self.cases)

    def freeze_dataset(self, dataset_id, content_sha256):
        self.frozen.append((dataset_id, content_sha256))


def _case() -> ReviewedEvalCase:
    return ReviewedEvalCase(
        case_code="ACTION_CLARIFY_001",
        question="我的订单怎么取消？",
        case_type=EvalCaseType.MULTI_CONDITION,
        expected_action=ExpectedAction.CLARIFY,
        dataset_split=DatasetSplit.DEVELOPMENT,
        required_fact_count=0,
        generation_metadata={
            "clarification_contract": {
                "missing_condition_key": "order_status",
                "missing_condition_label": "订单状态",
                "clarification_question": (
                    "订单目前是待支付、待出库还是已发货？"
                ),
                "acceptable_question_keywords": [
                    "订单状态",
                    "待支付",
                    "已发货",
                ],
                "branches": [
                    {
                        "condition_value": "PENDING_PAYMENT",
                        "label": "待支付",
                        "chunk_ids": [50],
                        "expected_outcome": "可以直接取消。",
                    },
                    {
                        "condition_value": "SHIPPED",
                        "label": "已发货",
                        "chunk_ids": [54],
                        "expected_outcome": "需要走售后流程。",
                    },
                ],
            }
        },
        evidences=[],
        review_status=ReviewStatus.PASSED,
        review_score=1.0,
        review_reason="人工按澄清契约审核通过",
    )


def test_register_action_calibration_freezes_canonical_dataset(
    tmp_path,
) -> None:
    input_path = tmp_path / "action.jsonl"
    output_path = tmp_path / "action.frozen.jsonl"
    write_case_jsonl(input_path, [_case()])
    repository = FakeDatasetRepository()

    result = register_action_calibration(
        args=SimpleNamespace(
            input=input_path,
            output=output_path,
            dataset_code="rag_eval_action",
            version="v1",
            name="RAG动作决策校准集",
        ),
        repository=repository,
        active_chunk_ids={50, 54},
    )

    assert result["dataset_id"] == 12
    assert result["case_count"] == 1
    assert len(result["content_sha256"]) == 64
    assert repository.created[0]["status"].value == "REVIEWED"
    assert repository.cases[0][0] == 12
    assert repository.frozen == [(12, result["content_sha256"])]
    assert output_path.exists()
