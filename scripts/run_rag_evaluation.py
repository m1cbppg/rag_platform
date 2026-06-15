import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.application.chat_service import ChatService  # noqa: E402
from src.rag_platform.application.clarification_policy_service import (  # noqa: E402
    CLARIFICATION_POLICY_VERSION,
    clarification_policy_snapshot,
)
from src.rag_platform.application.evidence_constraint_service import (  # noqa: E402
    EVIDENCE_CONSTRAINT_GUARD_VERSION,
)
from src.rag_platform.core.config import get_settings  # noqa: E402
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.experiment_runner import (  # noqa: E402
    ExperimentCase,
    ExperimentRunner,
)
from src.rag_platform.evaluation.judge_service import (  # noqa: E402
    ANSWER_JUDGE_PROMPT_VERSION,
    AnswerJudgeService,
)
from src.rag_platform.evaluation.models import (  # noqa: E402
    DatasetSplit,
    DatasetStatus,
    EvalCaseType,
    EvalRunConfig,
    EvidenceSpec,
    ExpectedAction,
    ReviewedEvalCase,
)
from src.rag_platform.evaluation.rag_adapter import (  # noqa: E402
    ChatServiceEvaluationAdapter,
)
from src.rag_platform.infrastructure.dashscope_chat import (  # noqa: E402
    DashScopeChatClient,
)
from src.rag_platform.rag.retrieval.business_domain import (  # noqa: E402
    BUSINESS_DOMAIN_ALIAS_VERSION,
    business_domain_alias_snapshot,
)
from src.rag_platform.rag.answer.action_decision_prompt import (  # noqa: E402
    ANSWER_ACTION_DECISION_PROMPT_VERSION,
)
from src.rag_platform.rag.adaptive.models import (  # noqa: E402
    ADAPTIVE_RETRIEVAL_POLICY_VERSION,
)
from src.rag_platform.rag.adaptive.query_rewrite_prompt import (  # noqa: E402
    QUERY_REWRITE_PROMPT_VERSION,
)
from src.rag_platform.rag.adaptive.query_decomposition_prompt import (  # noqa: E402
    QUERY_DECOMPOSITION_PROMPT_VERSION,
)
from src.rag_platform.rag.adaptive.intermediate_fact_prompt import (  # noqa: E402
    INTERMEDIATE_FACT_PROMPT_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行冻结数据集上的RAG自动化评测",
    )
    parser.add_argument(
        "--dataset",
        default="rag_eval_ecommerce:v1",
        help="数据集引用，格式为code:version",
    )
    parser.add_argument(
        "--split",
        choices=["development", "validation", "test"],
        default="development",
    )
    parser.add_argument("--experiment-version", default="V0")
    parser.add_argument(
        "--experiment-name",
        default="baseline-hybrid-rrf-rerank",
    )
    parser.add_argument("--run-code")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--adaptive-retrieval",
        choices=["enabled", "disabled"],
        help=(
            "仅覆盖本次评测进程的自适应检索开关，"
            "不修改.env；未指定时使用项目配置"
        ),
    )
    parser.add_argument(
        "--query-analysis",
        choices=["llm", "rule"],
        help=(
            "仅覆盖本次评测的Query分析方式；"
            "受控检索A/B建议使用rule固定上游Query计划"
        ),
    )
    parser.add_argument(
        "--query-decomposition",
        choices=["enabled", "disabled"],
        help=(
            "仅覆盖本次评测进程的复杂查询分解开关，"
            "不修改.env；未指定时使用项目配置"
        ),
    )
    parser.add_argument(
        "--expected-action",
        dest="expected_actions",
        action="append",
        choices=["answer", "refuse", "clarify"],
        help="只运行指定预期动作，可重复传入",
    )
    parser.add_argument(
        "--case-code",
        dest="case_codes",
        action="append",
        help="只运行指定case_code，可重复传入",
    )
    parser.add_argument(
        "--case-type",
        dest="case_types",
        action="append",
        choices=[
            "direct",
            "paraphrase",
            "exact",
            "multi_condition",
            "multi_hop",
            "conflict",
            "no_answer",
        ],
        help="只运行指定题型，可重复传入",
    )
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=Path("evaluation/prompts/answer_judge.txt"),
    )
    return parser.parse_args()


def parse_dataset_reference(value: str) -> tuple[str, str]:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError("数据集参数必须使用code:version格式")
    return parts[0].strip(), parts[1].strip()


def apply_runtime_overrides(settings, args):
    adaptive_retrieval = getattr(
        args,
        "adaptive_retrieval",
        None,
    )
    if adaptive_retrieval is not None:
        settings.adaptive_retrieval_enabled = (
            adaptive_retrieval == "enabled"
        )
    query_decomposition = getattr(
        args,
        "query_decomposition",
        None,
    )
    if query_decomposition is not None:
        settings.query_decomposition_enabled = (
            query_decomposition == "enabled"
        )
    query_analysis = getattr(args, "query_analysis", None)
    if query_analysis is not None:
        settings.query_analysis_use_llm = query_analysis == "llm"
    return settings


def build_config_snapshot(
    *,
    settings,
    args,
    dataset_sha256: str,
    git_dirty: bool,
) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "dataset_split": args.split,
        "limit": args.limit,
        "expected_actions": [
            value.upper()
            for value in (getattr(args, "expected_actions", None) or [])
        ],
        "case_codes": list(
            getattr(args, "case_codes", None) or []
        ),
        "case_types": [
            value.upper()
            for value in (
                getattr(args, "case_types", None) or []
            )
        ],
        "concurrency": args.concurrency,
        "top_k": args.top_k,
        "elasticsearch_index": settings.es_chunk_index,
        "milvus_collection": settings.milvus_collection,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "answer_model": settings.answer_model,
        "judge_model": settings.qwen_judge_model,
        "hybrid_fusion_method": settings.hybrid_fusion_method,
        "rrf_rank_constant": settings.rrf_rank_constant,
        "rrf_window_size": settings.rrf_window_size,
        "es_bm25_top_k": settings.es_bm25_top_k,
        "hybrid_final_top_k": settings.hybrid_final_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "context_max_tokens": settings.context_max_tokens,
        "context_max_chunks": settings.context_max_chunks,
        "query_analysis_use_llm": (
            settings.query_analysis_use_llm
        ),
        "query_analysis_cli_override": getattr(
            args,
            "query_analysis",
            None,
        ),
        "adaptive_retrieval_enabled": (
            settings.adaptive_retrieval_enabled
        ),
        "adaptive_retrieval_cli_override": getattr(
            args,
            "adaptive_retrieval",
            None,
        ),
        "adaptive_retrieval_policy_version": (
            ADAPTIVE_RETRIEVAL_POLICY_VERSION
        ),
        "adaptive_max_rounds": settings.adaptive_max_rounds,
        "adaptive_quality_good_threshold": (
            settings.adaptive_quality_good_threshold
        ),
        "adaptive_quality_poor_threshold": (
            settings.adaptive_quality_poor_threshold
        ),
        "adaptive_rerank_top1_threshold": (
            settings.adaptive_rerank_top1_threshold
        ),
        "adaptive_rerank_top3_threshold": (
            settings.adaptive_rerank_top3_threshold
        ),
        "adaptive_min_candidate_count": (
            settings.adaptive_min_candidate_count
        ),
        "adaptive_min_distinct_documents": (
            settings.adaptive_min_distinct_documents
        ),
        "adaptive_min_version_count": (
            settings.adaptive_min_version_count
        ),
        "adaptive_rewrite_model": settings.adaptive_rewrite_model,
        "adaptive_rewrite_max_attempts": (
            settings.adaptive_rewrite_max_attempts
        ),
        "adaptive_query_rewrite_prompt_version": (
            QUERY_REWRITE_PROMPT_VERSION
        ),
        "query_decomposition_enabled": (
            settings.query_decomposition_enabled
        ),
        "query_decomposition_cli_override": getattr(
            args,
            "query_decomposition",
            None,
        ),
        "query_decomposition_prompt_version": (
            QUERY_DECOMPOSITION_PROMPT_VERSION
        ),
        "query_decomposition_model": (
            settings.query_decomposition_model
        ),
        "query_decomposition_max_sub_queries": (
            settings.query_decomposition_max_sub_queries
        ),
        "query_decomposition_max_attempts": (
            settings.query_decomposition_max_attempts
        ),
        "query_decomposition_min_query_length": (
            settings.query_decomposition_min_query_length
        ),
        "query_decomposition_min_benefit_score": (
            settings.query_decomposition_min_benefit_score
        ),
        "query_decomposition_allow_dependent": (
            settings.query_decomposition_allow_dependent
        ),
        "query_decomposition_rerank_extra_limit": (
            settings.query_decomposition_rerank_extra_limit
        ),
        "sub_query_min_candidates": (
            settings.sub_query_min_candidates
        ),
        "sub_query_rerank_quota": (
            settings.sub_query_rerank_quota
        ),
        "dependent_multi_hop_enabled": (
            settings.dependent_multi_hop_enabled
        ),
        "dependent_multi_hop_max_hops": (
            settings.dependent_multi_hop_max_hops
        ),
        "dependent_fact_model": settings.dependent_fact_model,
        "dependent_fact_min_confidence": (
            settings.dependent_fact_min_confidence
        ),
        "dependent_fact_max_candidates": (
            settings.dependent_fact_max_candidates
        ),
        "dependent_fact_max_attempts": (
            settings.dependent_fact_max_attempts
        ),
        "dependent_fact_prompt_version": (
            INTERMEDIATE_FACT_PROMPT_VERSION
        ),
        "judge_prompt_version": ANSWER_JUDGE_PROMPT_VERSION,
        "clarification_policy_version": CLARIFICATION_POLICY_VERSION,
        "clarification_policies": clarification_policy_snapshot(),
        "evidence_constraint_guard_version": (
            EVIDENCE_CONSTRAINT_GUARD_VERSION
        ),
        "business_domain_alias_version": BUSINESS_DOMAIN_ALIAS_VERSION,
        "business_domain_aliases": business_domain_alias_snapshot(),
        "action_decision_enabled": settings.action_decision_enabled,
        "action_decision_model": settings.action_decision_model,
        "action_decision_prompt_version": (
            ANSWER_ACTION_DECISION_PROMPT_VERSION
        ),
        "action_decision_clarify_threshold": (
            settings.action_decision_clarify_threshold
        ),
        "action_decision_refuse_threshold": (
            settings.action_decision_refuse_threshold
        ),
        "action_decision_max_attempts": (
            settings.action_decision_max_attempts
        ),
        "git_dirty": git_dirty,
    }


def load_experiment_cases(
    *,
    repository: DatasetRepository,
    dataset_id: int,
    split: DatasetSplit,
    expected_actions: list[str] | None = None,
    case_codes: list[str] | None = None,
    case_types: list[str] | None = None,
    limit: int | None = None,
) -> list[ExperimentCase]:
    rows = repository.list_reviewed_cases(dataset_id, split)
    if case_codes:
        allowed_case_codes = set(case_codes)
        rows = [
            row
            for row in rows
            if str(row["case_code"]) in allowed_case_codes
        ]
    if expected_actions:
        allowed_actions = {
            ExpectedAction(value.upper()).value
            for value in expected_actions
        }
        rows = [
            row
            for row in rows
            if (
                row["expected_action"].value
                if isinstance(row["expected_action"], ExpectedAction)
                else str(row["expected_action"])
            )
            in allowed_actions
        ]
    if case_types:
        allowed_case_types = {
            EvalCaseType(value.upper()).value
            for value in case_types
        }
        rows = [
            row
            for row in rows
            if (
                row["case_type"].value
                if isinstance(row["case_type"], EvalCaseType)
                else str(row["case_type"])
            )
            in allowed_case_types
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit必须大于0")
        rows = rows[:limit]
    result = []
    for row in rows:
        case_id = int(row["id"])
        evidences = [
            EvidenceSpec.model_validate(item)
            for item in repository.list_case_evidence(case_id)
        ]
        result.append(
            ExperimentCase(
                case_id=case_id,
                case=ReviewedEvalCase.model_validate(
                    {
                        **row,
                        "evidences": evidences,
                    }
                ),
            )
        )
    return result


async def run(args: argparse.Namespace) -> int:
    if not 1 <= args.concurrency <= 3:
        raise ValueError("concurrency必须在1到3之间")
    dataset_code, version = parse_dataset_reference(args.dataset)
    split = DatasetSplit(args.split.upper())
    repository = DatasetRepository()
    dataset = repository.find_dataset(dataset_code, version)
    if dataset is None:
        raise ValueError(f"评测数据集不存在：{args.dataset}")
    if dataset["status"] != DatasetStatus.FROZEN.value:
        raise ValueError("只有FROZEN评测数据集可以运行实验")
    if not dataset.get("content_sha256"):
        raise ValueError("冻结数据集缺少content_sha256")

    cases = load_experiment_cases(
        repository=repository,
        dataset_id=int(dataset["id"]),
        split=split,
        expected_actions=args.expected_actions,
        case_codes=args.case_codes,
        case_types=args.case_types,
        limit=args.limit,
    )
    if not cases:
        raise ValueError("指定分片没有可执行评测题")

    settings = apply_runtime_overrides(get_settings(), args)
    git_sha = _git_output(["git", "rev-parse", "HEAD"])
    git_dirty = bool(_git_output(["git", "status", "--porcelain"]))
    run_code = args.run_code or (
        f"{args.experiment_version}_{split.value}_"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    existing = repository.find_run_by_code(run_code)
    if existing is None:
        config_snapshot = build_config_snapshot(
            settings=settings,
            args=args,
            dataset_sha256=dataset["content_sha256"],
            git_dirty=git_dirty,
        )
        run_id = repository.create_run(
            EvalRunConfig(
                run_code=run_code,
                dataset_id=int(dataset["id"]),
                experiment_version=args.experiment_version,
                experiment_name=args.experiment_name,
                git_commit_sha=git_sha or None,
                retrieval_mode=settings.default_retrieval_mode,
                embedding_model=settings.embedding_model,
                rerank_model=settings.rerank_model,
                answer_model=settings.answer_model,
                judge_model=settings.qwen_judge_model,
                config=config_snapshot,
                total_cases=len(cases),
            )
        )
    else:
        if int(existing["dataset_id"]) != int(dataset["id"]):
            raise ValueError("run_code已属于其他评测数据集")
        run_id = int(existing["id"])

    qwen = DashScopeChatClient(settings=settings)
    judge = AnswerJudgeService(
        client=qwen,
        prompt_template=args.judge_prompt.read_text(encoding="utf-8"),
        model=settings.qwen_judge_model,
    )
    runner = ExperimentRunner(
        repository=repository,
        rag_adapter=ChatServiceEvaluationAdapter(ChatService()),
        judge_service=judge,
        concurrency=args.concurrency,
        top_k=args.top_k,
    )
    try:
        summary = await runner.run(run_id=run_id, cases=cases)
    finally:
        await qwen.aclose()

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_code": run_code,
                "dataset": args.dataset,
                "split": split.value,
                "summary": summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _git_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
