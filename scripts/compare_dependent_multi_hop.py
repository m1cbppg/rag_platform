import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
from src.rag_platform.evaluation.dependent_multi_hop_comparison import (  # noqa: E402
    build_dependent_multi_hop_comparison,
    render_dependent_multi_hop_comparison_markdown,
)
from src.rag_platform.evaluation.metric_calculator import (  # noqa: E402
    GoldAnnotations,
    calculate_case_metrics,
)
from src.rag_platform.evaluation.models import ActualAction  # noqa: E402
from src.rag_platform.rag.adaptive.intermediate_fact_prompt import (  # noqa: E402
    INTERMEDIATE_FACT_PROMPT_VERSION,
)
from src.rag_platform.rag.adaptive.query_decomposition_prompt import (  # noqa: E402
    QUERY_DECOMPOSITION_PROMPT_VERSION,
)
from src.rag_platform.rag.graph.rag_retrieval_graph import (  # noqa: E402
    RagRetrievalGraphBuilder,
)
from src.rag_platform.schemas.rag_workflow import (  # noqa: E402
    RagRetrievalWorkflowRequest,
)


class GoldEvidence(BaseModel):
    hop: int = Field(ge=1, le=2)
    chunk_id: int = Field(gt=0)
    fact_key: str = Field(min_length=1)
    relevance_grade: int = Field(default=3, ge=1, le=3)


class DependentEvaluationCase(BaseModel):
    case_code: str = Field(min_length=1)
    chain_type: str = Field(min_length=1)
    question: str = Field(min_length=10)
    business_domain: str = "ecommerce_after_sales"
    target_doc_types: list[str] = Field(default_factory=list)
    source_case_codes: list[str] = Field(default_factory=list)
    expected_intermediate_fact_aliases: list[str] = Field(
        min_length=1
    )
    gold_evidences: list[GoldEvidence] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_two_hops(self):
        hops = {item.hop for item in self.gold_evidences}
        if hops != {1, 2}:
            raise ValueError("专项Case必须同时标注第一跳和第二跳Gold")
        return self

    @property
    def first_hop_gold_chunk_ids(self) -> set[int]:
        return {
            item.chunk_id
            for item in self.gold_evidences
            if item.hop == 1
        }

    @property
    def second_hop_gold_chunk_ids(self) -> set[int]:
        return {
            item.chunk_id
            for item in self.gold_evidences
            if item.hop == 2
        }

    def build_gold(self) -> GoldAnnotations:
        relevance_by_chunk: dict[int, int] = {}
        fact_keys_by_chunk: dict[int, set[str]] = {}
        for item in self.gold_evidences:
            relevance_by_chunk[item.chunk_id] = max(
                relevance_by_chunk.get(item.chunk_id, 0),
                item.relevance_grade,
            )
            fact_keys_by_chunk.setdefault(
                item.chunk_id,
                set(),
            ).add(item.fact_key)
        return GoldAnnotations(
            relevance_by_chunk=relevance_by_chunk,
            fact_keys_by_chunk=fact_keys_by_chunk,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="成对比较关闭/开启顺序依赖两跳检索",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(
            "evaluation/datasets/"
            "rag_dependent_multi_hop_v1.jsonl"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
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
            "m10_2_dependent_multi_hop_comparison.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "evaluation/reports/"
            "m10_2_dependent_multi_hop_comparison.md"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "evaluation/reports/"
            "m10_2_dependent_multi_hop_comparison.partial.jsonl"
        ),
    )
    return parser.parse_args()


def build_benchmark_settings(
    base: Settings,
    *,
    dependent_enabled: bool,
) -> Settings:
    return base.model_copy(
        update={
            "query_analysis_use_llm": False,
            "adaptive_retrieval_enabled": True,
            "adaptive_max_rounds": 2,
            "query_decomposition_enabled": True,
            "query_decomposition_allow_dependent": (
                dependent_enabled
            ),
            "dependent_multi_hop_enabled": dependent_enabled,
            "dependent_multi_hop_max_hops": 2,
        },
        deep=True,
    )


def load_dependent_cases(
    path: Path,
) -> list[DependentEvaluationCase]:
    if not path.exists():
        raise ValueError(f"专项数据集不存在：{path}")
    cases = [
        DependentEvaluationCase.model_validate_json(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    codes = [item.case_code for item in cases]
    if len(codes) != len(set(codes)):
        raise ValueError("专项数据集Case Code重复")
    return cases


def build_workflow_snapshot(
    *,
    workflow,
    case: DependentEvaluationCase,
    latency_ms: int,
) -> dict[str, Any]:
    gold = case.build_gold()
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
    attempts = list(workflow.retrieval_attempts or [])
    first_hop_chunk_ids = (
        _attempt_chunk_ids(attempts[0]) if attempts else []
    )
    dependent_attempts = [
        item
        for item in attempts
        if item.get("query_variant") == "DEPENDENT_HOP"
        or item.get("strategy") == "DEPENDENT_HOP"
    ]
    second_hop_chunk_ids = list(
        dict.fromkeys(
            chunk_id
            for attempt in dependent_attempts
            for chunk_id in _attempt_chunk_ids(attempt)
        )
    )

    decomposition = dict(workflow.decomposition or {})
    dependent_hop = dict(workflow.dependent_hop or {})
    intermediate_fact = str(
        dependent_hop.get("intermediate_fact") or ""
    ).strip()
    evidence_quote = str(
        dependent_hop.get("evidence_quote") or ""
    ).strip()
    supporting_chunk_id = dependent_hop.get(
        "supporting_chunk_id"
    )
    second_hop_query = str(
        dependent_hop.get("second_hop_query") or ""
    ).strip()
    fallback_used = bool(
        dependent_hop.get("fallback_used")
    )
    dependent_triggered = (
        decomposition.get("requires_decomposition") is True
        and decomposition.get("decomposition_type")
        == "DEPENDENT"
    )
    extraction_success = bool(
        intermediate_fact
        and evidence_quote
        and supporting_chunk_id is not None
        and not fallback_used
    )
    supporting_chunk_accurate = (
        supporting_chunk_id is not None
        and int(supporting_chunk_id)
        in case.first_hop_gold_chunk_ids
    )
    intermediate_fact_accurate = (
        extraction_success
        and _matches_alias(
            f"{intermediate_fact} {evidence_quote}",
            case.expected_intermediate_fact_aliases,
        )
    )
    second_query_contains_fact = bool(
        intermediate_fact
        and _normalize(intermediate_fact)
        in _normalize(second_hop_query)
    )
    first_hop_gold_hit = bool(
        set(first_hop_chunk_ids)
        & case.first_hop_gold_chunk_ids
    )
    second_hop_gold_hit = bool(
        set(second_hop_chunk_ids)
        & case.second_hop_gold_chunk_ids
    )
    fact_coverage = metrics.fact_coverage
    end_to_end_chain_success = bool(
        dependent_triggered
        and first_hop_gold_hit
        and extraction_success
        and supporting_chunk_accurate
        and intermediate_fact_accurate
        and second_query_contains_fact
        and second_hop_gold_hit
        and fact_coverage is not None
        and fact_coverage >= 1.0 - 1e-9
    )

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
        "first_hop_attempt_chunk_ids": first_hop_chunk_ids,
        "second_hop_attempt_chunk_ids": second_hop_chunk_ids,
        "retrieval_rounds": int(workflow.retrieval_round),
        "latency_ms": latency_ms,
        "decomposition": decomposition,
        "dependent_hop": dependent_hop,
        "retrieval_quality": workflow.retrieval_quality,
        "upstream_query_plan": _upstream_query_plan(workflow),
        "dependent_triggered": dependent_triggered,
        "first_hop_gold_hit": first_hop_gold_hit,
        "extraction_success": extraction_success,
        "supporting_chunk_accurate": (
            supporting_chunk_accurate
        ),
        "intermediate_fact_accurate": (
            intermediate_fact_accurate
        ),
        "second_query_contains_fact": (
            second_query_contains_fact
        ),
        "second_hop_gold_hit": second_hop_gold_hit,
        "fallback_used": fallback_used,
        "end_to_end_chain_success": (
            end_to_end_chain_success
        ),
    }


async def run(args: argparse.Namespace) -> int:
    if not 1 <= args.concurrency <= 3:
        raise ValueError("concurrency必须在1到3之间")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit必须大于0")

    cases = load_dependent_cases(args.dataset_path)
    if args.case_codes:
        selected_codes = set(args.case_codes)
        cases = [
            item
            for item in cases
            if item.case_code in selected_codes
        ]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("没有可执行的顺序依赖专项Case")

    base_settings = get_settings()
    settings_by_side = {
        "control": build_benchmark_settings(
            base_settings,
            dependent_enabled=False,
        ),
        "dependent": build_benchmark_settings(
            base_settings,
            dependent_enabled=True,
        ),
    }
    services = {
        side: _build_workflow_service(settings)
        for side, settings in settings_by_side.items()
    }

    completed_pairs = _load_checkpoint(args.checkpoint)
    completed_codes = {
        str(item["case_code"]) for item in completed_pairs
    }
    pending = [
        item
        for item in cases
        if item.case_code not in completed_codes
    ]
    errors: list[dict[str, str]] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    checkpoint_lock = asyncio.Lock()

    async def execute(case: DependentEvaluationCase) -> None:
        async with semaphore:
            try:
                pair = await _run_pair(
                    case=case,
                    services=services,
                    top_k=args.top_k,
                )
            except Exception as exc:
                errors.append(
                    {
                        "case_code": case.case_code,
                        "error": str(exc).strip()
                        or type(exc).__name__,
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
                dependent = pair["dependent"]
                print(
                    f"[{len(completed_pairs)}/{len(cases)}] "
                    f"{case.case_code} "
                    f"triggered={dependent['dependent_triggered']} "
                    f"fact={dependent['intermediate_fact_accurate']} "
                    f"hop2={dependent['second_hop_gold_hit']} "
                    f"coverage="
                    f"{dependent['metrics']['fact_coverage']}",
                    flush=True,
                )

    await asyncio.gather(*(execute(case) for case in pending))
    if errors:
        raise RuntimeError(
            "专项评测存在失败Case："
            + "；".join(
                f"{item['case_code']}={item['error']}"
                for item in errors
            )
        )

    dataset_bytes = args.dataset_path.read_bytes()
    report = build_dependent_multi_hop_comparison(
        pairs=completed_pairs,
        metadata={
            "dataset_path": args.dataset_path.as_posix(),
            "dataset_sha256": hashlib.sha256(
                dataset_bytes
            ).hexdigest(),
            "top_k": args.top_k,
            "concurrency": args.concurrency,
            "query_analysis_mode": "rule",
            "adaptive_retrieval_enabled": True,
            "control_dependent_enabled": False,
            "experiment_dependent_enabled": True,
            "query_decomposition_prompt_version": (
                QUERY_DECOMPOSITION_PROMPT_VERSION
            ),
            "intermediate_fact_prompt_version": (
                INTERMEDIATE_FACT_PROMPT_VERSION
            ),
            "query_decomposition_model": (
                settings_by_side[
                    "dependent"
                ].query_decomposition_model
            ),
            "dependent_fact_model": (
                settings_by_side[
                    "dependent"
                ].dependent_fact_model
            ),
            "dependent_fact_min_confidence": (
                settings_by_side[
                    "dependent"
                ].dependent_fact_min_confidence
            ),
            "git_commit_sha": _git_output(
                ["git", "rev-parse", "HEAD"]
            ),
            "git_dirty": bool(
                _git_output(["git", "status", "--porcelain"])
            ),
            "generated_at": (
                datetime.now(timezone.utc).isoformat()
            ),
        },
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
    case: DependentEvaluationCase,
    services: dict[str, RagWorkflowService],
    top_k: int,
) -> dict[str, Any]:
    results = {}
    order = pair_execution_order(case.case_code)
    for side in order:
        results[side] = await _run_side(
            service=services[side],
            case=case,
            side=side,
            top_k=top_k,
        )
    return {
        "case_code": case.case_code,
        "chain_type": case.chain_type,
        "question": case.question,
        "source_case_codes": case.source_case_codes,
        "expected_intermediate_fact_aliases": (
            case.expected_intermediate_fact_aliases
        ),
        "first_hop_gold_chunk_ids": sorted(
            case.first_hop_gold_chunk_ids
        ),
        "second_hop_gold_chunk_ids": sorted(
            case.second_hop_gold_chunk_ids
        ),
        "execution_order": list(order),
        "upstream_query_plan_match": (
            results["control"]["upstream_query_plan"]
            == results["dependent"]["upstream_query_plan"]
        ),
        "control": results["control"],
        "dependent": results["dependent"],
    }


def pair_execution_order(
    case_code: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(case_code.encode("utf-8")).digest()
    if digest[0] % 2 == 0:
        return "control", "dependent"
    return "dependent", "control"


async def _run_side(
    *,
    service: RagWorkflowService,
    case: DependentEvaluationCase,
    side: str,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    workflow = await service.run_retrieval_workflow(
        RagRetrievalWorkflowRequest(
            question=case.question,
            session_id=(
                f"m10-2-{side}-{case.case_code}"
            ),
            business_domain=case.business_domain,
            top_k=top_k,
        )
    )
    return build_workflow_snapshot(
        workflow=workflow,
        case=case,
        latency_ms=int(
            (time.perf_counter() - started) * 1000
        ),
    )


def _build_workflow_service(
    settings: Settings,
) -> RagWorkflowService:
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


def _attempt_chunk_ids(attempt: dict[str, Any]) -> list[int]:
    documents = (
        attempt.get("reranked_documents")
        or attempt.get("documents")
        or []
    )
    return list(
        dict.fromkeys(
            int(item["chunk_id"])
            for item in documents
            if item.get("chunk_id") is not None
        )
    )


def _matches_alias(
    value: str,
    aliases: list[str],
) -> bool:
    normalized = _normalize(value)
    return any(
        alias_normalized in normalized
        or (
            len(normalized) >= 2
            and normalized in alias_normalized
        )
        for alias in aliases
        if (alias_normalized := _normalize(alias))
    )


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).lower()


def _upstream_query_plan(workflow) -> dict[str, Any]:
    analysis = workflow.query_analysis or {}
    return {
        "rewritten_query": analysis.get("rewritten_query"),
        "expanded_queries": analysis.get("expanded_queries") or [],
        "target_doc_types": analysis.get("target_doc_types") or [],
        "retrieval_mode": analysis.get("retrieval_mode"),
        "business_domain": analysis.get("business_domain"),
    }


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
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
        render_dependent_multi_hop_comparison_markdown(
            report
        ),
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
