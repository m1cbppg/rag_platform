from src.rag_platform.evaluation.experiment_runner import (
    ExperimentCase,
    ExperimentRunner,
)
from src.rag_platform.evaluation.models import (
    ActualAction,
    DatasetSplit,
    EvalCaseType,
    EvalRunStatus,
    EvidenceSpec,
    ExpectedAction,
    JudgeScore,
    MappingStatus,
    ReviewStatus,
    ReviewedEvalCase,
)
from src.rag_platform.evaluation.rag_adapter import (
    RagEvaluationObservation,
)


def _case(case_id: int, code: str, question: str) -> ExperimentCase:
    return ExperimentCase(
        case_id=case_id,
        case=ReviewedEvalCase(
            case_code=code,
            question=question,
            reference_answer="待支付订单可以直接取消。",
            case_type=EvalCaseType.DIRECT,
            expected_action=ExpectedAction.ANSWER,
            dataset_split=DatasetSplit.DEVELOPMENT,
            required_fact_count=1,
            business_domain="ecommerce_after_sales",
            evidences=[
                EvidenceSpec(
                    source_doc_code="FAQ_ORDER_STATUS_001",
                    evidence_quote="待支付订单可以直接取消。",
                    fact_key="cancel_pending",
                    relevance_grade=3,
                    mapped_doc_id=1,
                    mapped_chunk_id=101,
                    mapping_status=MappingStatus.MAPPED,
                )
            ],
            review_status=ReviewStatus.PASSED,
        ),
    )


class FakeRepository:
    def __init__(self, skip_case_ids: set[int] | None = None) -> None:
        self.skip_case_ids = skip_case_ids or set()
        self.results: dict[int, dict] = {}
        self.hits: list[tuple[int, list[dict]]] = []
        self.judges: list[tuple[int, JudgeScore]] = []
        self.finished_run: dict | None = None

    def start_run(self, run_id):
        self.run_id = run_id

    def prepare_case_result(self, run_id, case_id):
        result_id = case_id + 1000
        if case_id in self.skip_case_ids:
            return result_id, False
        return result_id, True

    def update_case_result_trace(self, case_result_id, trace_id):
        self.results.setdefault(case_result_id, {})["trace_id"] = trace_id

    def save_retrieval_hits(self, case_result_id, hits):
        self.hits.append((case_result_id, hits))

    def finish_case_result(self, case_result_id, **kwargs):
        self.results.setdefault(case_result_id, {}).update(kwargs)
        self.results[case_result_id]["status"] = (
            "FAILED"
            if kwargs.get("error_message") is not None
            else "SUCCESS"
        )

    def save_judge_result(self, case_result_id, score):
        self.judges.append((case_result_id, score))
        return len(self.judges)

    def list_run_case_results(self, run_id):
        rows = []
        for result_id, result in self.results.items():
            metrics = result.get("metrics")
            rows.append(
                {
                    "id": result_id,
                    "case_type": "DIRECT",
                    "expected_action": "ANSWER",
                    "actual_action": result.get(
                        "actual_action",
                        ActualAction.ANSWER,
                    ).value,
                    "status": result["status"],
                    "recall_at_5": (
                        metrics.recall_at_5 if metrics else None
                    ),
                    "fact_coverage": (
                        metrics.fact_coverage if metrics else None
                    ),
                    "latency_ms": result.get("latency_ms"),
                    "judge_passed": next(
                        (
                            int(score.passed)
                            for saved_id, score in self.judges
                            if saved_id == result_id
                        ),
                        None,
                    ),
                }
            )
        return rows

    def finish_run(self, **kwargs):
        self.finished_run = kwargs


class FakeRagAdapter:
    def __init__(self, failing_questions: set[str] | None = None) -> None:
        self.failing_questions = failing_questions or set()
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["question"] in self.failing_questions:
            raise RuntimeError("RAG调用失败")
        return RagEvaluationObservation(
            trace_id=f"trace-{len(self.calls)}",
            answer="待支付订单可以直接取消。[C1]",
            actual_action=ActualAction.ANSWER,
            retrieved_chunk_ids=[101],
            cited_chunk_ids=[101],
            citations=[{"citation_id": "C1", "chunk_id": 101}],
            context="[C1] 待支付订单可以直接取消。",
            context_blocks=["[C1] 待支付订单可以直接取消。"],
            retrieval_hits=[
                {
                    "query_text": kwargs["question"],
                    "channel": "FINAL",
                    "chunk_id": 101,
                    "rank_no": 1,
                }
            ],
            latency_ms=100,
        )


class FakeJudge:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def judge(self, **kwargs):
        self.calls.append(kwargs)
        return JudgeScore(
            judge_provider="fake",
            judge_model="fake-judge",
            judge_prompt_version="v1",
            faithfulness_score=1.0,
            answer_relevance_score=1.0,
            completeness_score=1.0,
            citation_entailment_score=1.0,
            passed=True,
            reason={},
        )


class EmptyMessageFailureAdapter:
    async def run(self, **kwargs):
        raise Exception()


async def test_runner_executes_case_without_leaking_reference_answer() -> None:
    repository = FakeRepository()
    adapter = FakeRagAdapter()
    judge = FakeJudge()

    summary = await ExperimentRunner(
        repository=repository,
        rag_adapter=adapter,
        judge_service=judge,
        concurrency=1,
        top_k=5,
    ).run(run_id=7, cases=[_case(1, "CASE_1", "怎么取消？")])

    assert adapter.calls == [
        {
            "question": "怎么取消？",
            "session_id": "eval-7-CASE_1",
            "business_domain": "ecommerce_after_sales",
            "top_k": 5,
        }
    ]
    assert "reference_answer" not in adapter.calls[0]
    assert repository.results[1001]["metrics"].recall_at_1 == 1.0
    assert repository.hits[0][1][0]["is_gold"] is True
    assert judge.calls[0]["case"].reference_answer is not None
    assert summary["counts"]["completed"] == 1
    assert repository.finished_run["status"] == EvalRunStatus.SUCCESS


async def test_runner_isolates_failure_and_marks_partial_run() -> None:
    repository = FakeRepository()
    adapter = FakeRagAdapter(failing_questions={"失败题"})

    summary = await ExperimentRunner(
        repository=repository,
        rag_adapter=adapter,
        judge_service=FakeJudge(),
        concurrency=2,
    ).run(
        run_id=8,
        cases=[
            _case(1, "CASE_1", "正常题"),
            _case(2, "CASE_2", "失败题"),
        ],
    )

    assert summary["counts"]["completed"] == 1
    assert summary["counts"]["failed"] == 1
    assert repository.results[1002]["actual_action"] == ActualAction.ERROR
    assert repository.finished_run["status"] == EvalRunStatus.PARTIAL


async def test_runner_skips_successful_case_when_resuming() -> None:
    repository = FakeRepository(skip_case_ids={1})
    adapter = FakeRagAdapter()

    await ExperimentRunner(
        repository=repository,
        rag_adapter=adapter,
        judge_service=FakeJudge(),
    ).run(run_id=9, cases=[_case(1, "CASE_1", "已完成题")])

    assert adapter.calls == []


async def test_runner_preserves_exception_type_when_message_is_empty() -> None:
    repository = FakeRepository()

    summary = await ExperimentRunner(
        repository=repository,
        rag_adapter=EmptyMessageFailureAdapter(),
        judge_service=FakeJudge(),
    ).run(run_id=10, cases=[_case(1, "CASE_1", "空异常题")])

    assert summary["counts"]["failed"] == 1
    assert repository.results[1001]["error_message"] == "Exception"
    assert repository.results[1001]["status"] == "FAILED"
