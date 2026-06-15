import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from src.rag_platform.evaluation.metric_calculator import (
    build_gold_annotations,
    calculate_case_metrics,
)
from src.rag_platform.evaluation.models import (
    ActualAction,
    EvalRunStatus,
    JudgeScore,
    ReviewedEvalCase,
)
from src.rag_platform.evaluation.rag_adapter import (
    RagEvaluationObservation,
)
from src.rag_platform.evaluation.run_summary import build_run_summary


@dataclass(frozen=True)
class ExperimentCase:
    case_id: int
    case: ReviewedEvalCase


class EvaluationRepository(Protocol):
    def start_run(self, run_id: int) -> None: ...

    def prepare_case_result(
        self,
        run_id: int,
        case_id: int,
    ) -> tuple[int, bool]: ...

    def update_case_result_trace(
        self,
        case_result_id: int,
        trace_id: str,
    ) -> None: ...

    def save_retrieval_hits(
        self,
        case_result_id: int,
        hits: list[dict[str, Any]],
    ) -> None: ...

    def finish_case_result(self, case_result_id: int, **kwargs) -> None: ...

    def save_judge_result(
        self,
        case_result_id: int,
        score: JudgeScore,
    ) -> int: ...

    def list_run_case_results(
        self,
        run_id: int,
    ) -> list[dict[str, Any]]: ...

    def finish_run(self, **kwargs) -> None: ...


class RagAdapter(Protocol):
    async def run(self, **kwargs) -> RagEvaluationObservation: ...


class JudgeService(Protocol):
    async def judge(self, **kwargs) -> JudgeScore: ...


class ExperimentRunner:
    def __init__(
        self,
        *,
        repository: EvaluationRepository,
        rag_adapter: RagAdapter,
        judge_service: JudgeService,
        concurrency: int = 1,
        top_k: int = 20,
    ) -> None:
        if not 1 <= concurrency <= 3:
            raise ValueError("评测并发数必须在1到3之间")
        self.repository = repository
        self.rag_adapter = rag_adapter
        self.judge_service = judge_service
        self.semaphore = asyncio.Semaphore(concurrency)
        self.top_k = top_k

    async def run(
        self,
        *,
        run_id: int,
        cases: list[ExperimentCase],
    ) -> dict[str, Any]:
        self.repository.start_run(run_id)
        await asyncio.gather(
            *[
                self._execute_case(run_id=run_id, experiment_case=item)
                for item in cases
            ]
        )
        rows = self.repository.list_run_case_results(run_id)
        summary = build_run_summary(rows)
        completed = summary["counts"]["completed"]
        failed = summary["counts"]["failed"]
        if failed == 0:
            status = EvalRunStatus.SUCCESS
        elif completed > 0:
            status = EvalRunStatus.PARTIAL
        else:
            status = EvalRunStatus.FAILED
        self.repository.finish_run(
            run_id=run_id,
            status=status,
            completed_cases=completed,
            failed_cases=failed,
            summary_metrics=summary,
            error_message=None,
        )
        return summary

    async def _execute_case(
        self,
        *,
        run_id: int,
        experiment_case: ExperimentCase,
    ) -> None:
        async with self.semaphore:
            case_result_id, should_run = (
                self.repository.prepare_case_result(
                    run_id,
                    experiment_case.case_id,
                )
            )
            if not should_run:
                return
            case = experiment_case.case
            gold = build_gold_annotations(case.evidences)
            observation: RagEvaluationObservation | None = None
            try:
                observation = await self.rag_adapter.run(
                    question=case.question,
                    session_id=(
                        f"eval-{run_id}-{case.case_code}"
                    ),
                    business_domain=case.business_domain,
                    top_k=self.top_k,
                )
                self.repository.update_case_result_trace(
                    case_result_id,
                    observation.trace_id,
                )
                hits = [
                    {
                        **hit,
                        "is_gold": (
                            int(hit["chunk_id"])
                            in gold.relevance_by_chunk
                        ),
                    }
                    for hit in observation.retrieval_hits
                ]
                self.repository.save_retrieval_hits(
                    case_result_id,
                    hits,
                )
                metrics = calculate_case_metrics(
                    retrieved_chunk_ids=observation.retrieved_chunk_ids,
                    relevance_by_chunk=gold.relevance_by_chunk,
                    fact_keys_by_chunk=gold.fact_keys_by_chunk,
                    cited_chunk_ids=observation.cited_chunk_ids,
                    expected_action=case.expected_action,
                    actual_action=observation.actual_action,
                    retrieval_rounds=observation.retrieval_rounds,
                )
                judge_score = await self.judge_service.judge(
                    case=case,
                    system_answer=observation.answer,
                    actual_action=observation.actual_action,
                    context_blocks=observation.context_blocks,
                    citations=observation.citations,
                )
                self.repository.save_judge_result(
                    case_result_id,
                    judge_score,
                )
                self.repository.finish_case_result(
                    case_result_id=case_result_id,
                    actual_action=observation.actual_action,
                    generated_answer=observation.answer,
                    retrieved_chunk_ids=observation.retrieved_chunk_ids,
                    cited_chunk_ids=observation.cited_chunk_ids,
                    metrics=metrics,
                    latency_ms=observation.latency_ms,
                )
            except Exception as exc:
                error_message = str(exc).strip() or type(exc).__name__
                actual_action = (
                    observation.actual_action
                    if observation is not None
                    else ActualAction.ERROR
                )
                metrics = calculate_case_metrics(
                    retrieved_chunk_ids=(
                        observation.retrieved_chunk_ids
                        if observation is not None
                        else []
                    ),
                    relevance_by_chunk=gold.relevance_by_chunk,
                    fact_keys_by_chunk=gold.fact_keys_by_chunk,
                    cited_chunk_ids=(
                        observation.cited_chunk_ids
                        if observation is not None
                        else []
                    ),
                    expected_action=case.expected_action,
                    actual_action=actual_action,
                    retrieval_rounds=(
                        observation.retrieval_rounds
                        if observation is not None
                        else 1
                    ),
                )
                self.repository.finish_case_result(
                    case_result_id=case_result_id,
                    actual_action=actual_action,
                    generated_answer=(
                        observation.answer
                        if observation is not None
                        else None
                    ),
                    retrieved_chunk_ids=(
                        observation.retrieved_chunk_ids
                        if observation is not None
                        else []
                    ),
                    cited_chunk_ids=(
                        observation.cited_chunk_ids
                        if observation is not None
                        else []
                    ),
                    metrics=metrics,
                    latency_ms=(
                        observation.latency_ms
                        if observation is not None
                        else None
                    ),
                    error_message=error_message,
                )
