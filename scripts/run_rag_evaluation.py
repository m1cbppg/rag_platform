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
from src.rag_platform.core.config import get_settings  # noqa: E402
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)
from src.rag_platform.evaluation.experiment_runner import (  # noqa: E402
    ExperimentCase,
    ExperimentRunner,
)
from src.rag_platform.evaluation.judge_service import (  # noqa: E402
    AnswerJudgeService,
)
from src.rag_platform.evaluation.models import (  # noqa: E402
    DatasetSplit,
    DatasetStatus,
    EvalRunConfig,
    EvidenceSpec,
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
        "judge_prompt_version": "v1",
        "business_domain_alias_version": BUSINESS_DOMAIN_ALIAS_VERSION,
        "business_domain_aliases": business_domain_alias_snapshot(),
        "git_dirty": git_dirty,
    }


def load_experiment_cases(
    *,
    repository: DatasetRepository,
    dataset_id: int,
    split: DatasetSplit,
    limit: int | None = None,
) -> list[ExperimentCase]:
    rows = repository.list_reviewed_cases(dataset_id, split)
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
        limit=args.limit,
    )
    if not cases:
        raise ValueError("指定分片没有可执行评测题")

    settings = get_settings()
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
