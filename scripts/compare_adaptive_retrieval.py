import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.application.context_build_service import (  # noqa: E402
    ContextBuildService,
)
from src.rag_platform.application.query_understanding_service import (  # noqa: E402
    QueryUnderstandingService,
)
from src.rag_platform.application.rag_workflow_service import (  # noqa: E402
    RagWorkflowService,
)
from src.rag_platform.application.rerank_service import (  # noqa: E402
    RerankService,
)
from src.rag_platform.core.config import Settings, get_settings  # noqa: E402
from src.rag_platform.evaluation.adaptive_retrieval_comparison import (  # noqa: E402
    build_adaptive_retrieval_comparison,
    render_adaptive_retrieval_comparison_markdown,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.metric_calculator import (  # noqa: E402
    GoldAnnotations,
    build_gold_annotations,
    calculate_case_metrics,
)
from src.rag_platform.evaluation.models import (  # noqa: E402
    ActualAction,
    DatasetSplit,
    DatasetStatus,
)
from src.rag_platform.rag.adaptive.models import (  # noqa: E402
    ADAPTIVE_RETRIEVAL_POLICY_VERSION,
)
from src.rag_platform.rag.adaptive.query_rewrite_prompt import (  # noqa: E402
    QUERY_REWRITE_PROMPT_VERSION,
)
from src.rag_platform.rag.graph.rag_retrieval_graph import (  # noqa: E402
    RagRetrievalGraphBuilder,
)
from src.rag_platform.schemas.rag_workflow import (  # noqa: E402
    RagRetrievalWorkflowRequest,
)
from scripts.run_rag_evaluation import (  # noqa: E402
    load_experiment_cases,
    parse_dataset_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定Query计划，成对比较关闭/开启自适应检索",
    )
    parser.add_argument(
        "--dataset",
        default="rag_eval_ecommerce:v2",
    )
    parser.add_argument(
        "--split",
        choices=["development", "validation"],
        default="development",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-code",
        dest="case_codes",
        action="append",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "evaluation/reports/"
            "m9_adaptive_retrieval_comparison_v2.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "evaluation/reports/"
            "m9_adaptive_retrieval_comparison_v2.md"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "evaluation/reports/"
            "m9_adaptive_retrieval_comparison_v2.partial.jsonl"
        ),
    )
    return parser.parse_args()


def build_benchmark_settings(
    base: Settings,
    *,
    adaptive_enabled: bool,
) -> Settings:
    return base.model_copy(
        update={
            "query_analysis_use_llm": False,
            "adaptive_retrieval_enabled": adaptive_enabled,
        },
        deep=True,
    )


def build_workflow_snapshot(
    *,
    workflow,
    gold: GoldAnnotations,
    latency_ms: int,
) -> dict[str, Any]:
    retrieved_chunk_ids = list(
        dict.fromkeys(
            int(citation["chunk_id"])
            for citation in workflow.citations
            if citation.get("chunk_id") is not None
        )
    )
    metrics = calculate_case_metrics(
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevance_by_chunk=gold.relevance_by_chunk,
        fact_keys_by_chunk=gold.fact_keys_by_chunk,
        cited_chunk_ids=[],
        expected_action="ANSWER",
        actual_action=ActualAction.ANSWER,
        retrieval_rounds=int(workflow.retrieval_round),
    )
    attempts = [
        _attempt_snapshot(attempt)
        for attempt in workflow.retrieval_attempts
    ]
    retry_strategies = [
        str(attempt["strategy"])
        for attempt in attempts
        if int(attempt["round_no"]) > 1
    ]
    return {
        "metrics": {
            key: getattr(metrics, key)
            for key in (
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "reciprocal_rank",
                "ndcg_at_5",
                "ndcg_at_10",
                "fact_coverage",
            )
        },
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieval_rounds": int(workflow.retrieval_round),
        "latency_ms": latency_ms,
        "retry_strategies": retry_strategies,
        "query_plan": _query_plan(workflow, attempts),
        "retrieval_quality": workflow.retrieval_quality,
        "attempts": attempts,
    }


async def run(args: argparse.Namespace) -> int:
    if not 1 <= args.concurrency <= 3:
        raise ValueError("concurrency必须在1到3之间")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit必须大于0")

    dataset_code, version = parse_dataset_reference(args.dataset)
    split = DatasetSplit(args.split.upper())
    repository = DatasetRepository()
    dataset = repository.find_dataset(dataset_code, version)
    if dataset is None:
        raise ValueError(f"评测数据集不存在：{args.dataset}")
    if dataset["status"] != DatasetStatus.FROZEN.value:
        raise ValueError("只有FROZEN评测数据集可以运行专项评测")

    cases = load_experiment_cases(
        repository=repository,
        dataset_id=int(dataset["id"]),
        split=split,
        expected_actions=["answer"],
        case_codes=args.case_codes,
        limit=args.limit,
    )
    if not cases:
        raise ValueError("没有可执行的ANSWER评测题")

    base_settings = get_settings()
    control_settings = build_benchmark_settings(
        base_settings,
        adaptive_enabled=False,
    )
    adaptive_settings = build_benchmark_settings(
        base_settings,
        adaptive_enabled=True,
    )
    control_service = _build_workflow_service(control_settings)
    adaptive_service = _build_workflow_service(adaptive_settings)

    completed_pairs = _load_checkpoint(args.checkpoint)
    completed_codes = {
        str(item["case_code"])
        for item in completed_pairs
    }
    pending = [
        item
        for item in cases
        if item.case.case_code not in completed_codes
    ]
    errors: list[dict[str, str]] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    checkpoint_lock = asyncio.Lock()
    progress = {"completed": len(completed_pairs)}

    async def execute(experiment_case) -> None:
        async with semaphore:
            case = experiment_case.case
            try:
                pair = await _run_pair(
                    case=case,
                    control_service=control_service,
                    adaptive_service=adaptive_service,
                    top_k=args.top_k,
                )
            except Exception as exc:
                errors.append(
                    {
                        "case_code": case.case_code,
                        "error": str(exc).strip() or type(exc).__name__,
                    }
                )
                print(
                    f"[失败] {case.case_code}: "
                    f"{errors[-1]['error']}",
                    flush=True,
                )
                return
            async with checkpoint_lock:
                _append_checkpoint(args.checkpoint, pair)
                completed_pairs.append(pair)
                progress["completed"] += 1
                print(
                    f"[{progress['completed']}/{len(cases)}] "
                    f"{case.case_code} "
                    f"rounds={pair['adaptive']['retrieval_rounds']} "
                    f"fact_delta="
                    f"{pair['adaptive']['metrics']['fact_coverage'] - pair['control']['metrics']['fact_coverage']:+.3f}",
                    flush=True,
                )

    await asyncio.gather(*(execute(item) for item in pending))
    if errors:
        raise RuntimeError(
            "专项评测存在失败Case："
            + "；".join(
                f"{item['case_code']}={item['error']}"
                for item in errors
            )
        )

    metadata = {
        "dataset": args.dataset,
        "dataset_id": int(dataset["id"]),
        "dataset_sha256": dataset["content_sha256"],
        "split": split.value,
        "expected_action": "ANSWER",
        "query_analysis_mode": "rule",
        "pair_execution_order": "alternating_by_case_code_sha256",
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "adaptive_policy_version": (
            ADAPTIVE_RETRIEVAL_POLICY_VERSION
        ),
        "query_rewrite_prompt_version": (
            QUERY_REWRITE_PROMPT_VERSION
        ),
        "adaptive_max_rounds": (
            adaptive_settings.adaptive_max_rounds
        ),
        "adaptive_rerank_top1_threshold": (
            adaptive_settings.adaptive_rerank_top1_threshold
        ),
        "adaptive_rerank_top3_threshold": (
            adaptive_settings.adaptive_rerank_top3_threshold
        ),
        "git_commit_sha": _git_output(
            ["git", "rev-parse", "HEAD"]
        ),
        "git_dirty": bool(
            _git_output(["git", "status", "--porcelain"])
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report = build_adaptive_retrieval_comparison(
        pairs=completed_pairs,
        metadata=metadata,
    )
    _write_report(
        report=report,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    args.checkpoint.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "output_md": args.output_md.as_posix(),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


async def _run_pair(
    *,
    case,
    control_service: RagWorkflowService,
    adaptive_service: RagWorkflowService,
    top_k: int,
) -> dict[str, Any]:
    gold = build_gold_annotations(case.evidences)
    order = pair_execution_order(case.case_code)
    results = {}
    services = {
        "control": control_service,
        "adaptive": adaptive_service,
    }
    for side in order:
        results[side] = await _run_side(
            service=services[side],
            case=case,
            gold=gold,
            top_k=top_k,
        )
    control = results["control"]
    adaptive = results["adaptive"]
    return {
        "case_code": case.case_code,
        "case_type": case.case_type.value,
        "question": case.question,
        "execution_order": list(order),
        "initial_query_plan_match": (
            control["query_plan"] == adaptive["query_plan"]
        ),
        "control": control,
        "adaptive": adaptive,
    }


def pair_execution_order(case_code: str) -> tuple[str, str]:
    digest = hashlib.sha256(case_code.encode("utf-8")).digest()
    if digest[0] % 2 == 0:
        return "control", "adaptive"
    return "adaptive", "control"


async def _run_side(
    *,
    service: RagWorkflowService,
    case,
    gold: GoldAnnotations,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    workflow = await service.run_retrieval_workflow(
        RagRetrievalWorkflowRequest(
            question=case.question,
            session_id=f"m9-retrieval-{case.case_code}",
            business_domain=case.business_domain,
            top_k=top_k,
        )
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    return build_workflow_snapshot(
        workflow=workflow,
        gold=gold,
        latency_ms=latency_ms,
    )


def _build_workflow_service(settings: Settings) -> RagWorkflowService:
    query_service = QueryUnderstandingService(
        settings=settings,
        repository=_NoOpQueryAnalysisRepository(),
    )
    graph = RagRetrievalGraphBuilder(
        settings=settings,
        query_understanding_service=query_service,
        rerank_service=RerankService(settings=settings),
        context_build_service=ContextBuildService(
            settings=settings
        ),
    ).build()
    return RagWorkflowService(graph=graph)


def _attempt_snapshot(attempt: dict[str, Any]) -> dict[str, Any]:
    reranked = attempt.get("reranked_documents") or []
    return {
        "round_no": int(attempt.get("round_no") or 1),
        "strategy": attempt.get("strategy") or "INITIAL",
        "query_variant": (
            attempt.get("query_variant") or "ORIGINAL"
        ),
        "queries": list(attempt.get("queries") or []),
        "retrieval_mode": attempt.get("retrieval_mode"),
        "doc_type_filter": attempt.get("doc_type_filter"),
        "business_domain_filter": attempt.get(
            "business_domain_filter"
        ),
        "removed_filters": list(
            attempt.get("removed_filters") or []
        ),
        "quality": attempt.get("quality") or {},
        "top_reranked": [
            {
                "chunk_id": int(document["chunk_id"]),
                "score": float(
                    document.get("rerank_score")
                    or (document.get("metadata") or {}).get(
                        "rerank_score"
                    )
                    or 0.0
                ),
            }
            for document in reranked[:5]
            if document.get("chunk_id") is not None
        ],
    }


def _query_plan(
    workflow,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    first = attempts[0] if attempts else {}
    analysis = workflow.query_analysis or {}
    return {
        "rewritten_query": analysis.get("rewritten_query"),
        "expanded_queries": analysis.get("expanded_queries") or [],
        "target_doc_types": analysis.get("target_doc_types") or [],
        "retrieval_mode": analysis.get("retrieval_mode"),
        "business_domain": analysis.get("business_domain"),
        "queries": first.get("queries") or [],
        "doc_type_filter": first.get("doc_type_filter"),
        "business_domain_filter": first.get(
            "business_domain_filter"
        ),
    }


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_checkpoint(
    path: Path,
    pair: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                pair,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _write_report(
    *,
    report: dict[str, Any],
    output_json: Path,
    output_md: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        render_adaptive_retrieval_comparison_markdown(report),
        encoding="utf-8",
    )


def _git_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


class _NoOpQueryAnalysisRepository:
    def save_analysis_log(self, **kwargs) -> None:
        return None


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
