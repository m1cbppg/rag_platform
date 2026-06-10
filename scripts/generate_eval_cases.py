import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.case_persistence import (  # noqa: E402
    load_generated_case_jsonl,
    write_case_jsonl,
)
from src.rag_platform.evaluation.case_planner import (  # noqa: E402
    CaseSeedPlanner,
    load_case_context,
    load_case_plan,
)
from src.rag_platform.evaluation.case_services import (  # noqa: E402
    CaseGenerationService,
)
from src.rag_platform.evaluation.case_validation import (  # noqa: E402
    classify_grouped_semantic_duplicates,
    normalize_question,
    validate_generated_case,
)
from src.rag_platform.evaluation.models import (  # noqa: E402
    DatasetSplit,
    EvalCaseType,
)
from src.rag_platform.infrastructure.dashscope_embedding import (  # noqa: E402
    DashScopeEmbeddingClient,
)
from src.rag_platform.infrastructure.deepseek import DeepSeekClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行M4候选评测题生成和确定性去重",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("evaluation/blueprints/ecommerce_case_plan.json"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("evaluation/corpus/catalog.json"),
    )
    parser.add_argument(
        "--document-blueprint",
        type=Path,
        default=Path("evaluation/blueprints/ecommerce_document_plan.json"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("evaluation/prompts/case_generate.txt"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.generated.raw.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.generated.jsonl"),
    )
    parser.add_argument(
        "--supplement-split",
        choices=[item.value for item in DatasetSplit],
    )
    parser.add_argument(
        "--supplement-type",
        choices=[
            EvalCaseType.DIRECT.value,
            EvalCaseType.PARAPHRASE.value,
            EvalCaseType.EXACT.value,
        ],
    )
    parser.add_argument("--supplement-count", type=int, default=0)
    parser.add_argument("--supplement-round", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def _embed_questions(
    client: DashScopeEmbeddingClient,
    questions: list[str],
) -> list[list[float]]:
    vectors = []
    batch_size = client.settings.embedding_batch_size
    for start in range(0, len(questions), batch_size):
        vectors.extend(
            await client.embed_documents(
                questions[start:start + batch_size]
            )
        )
    return vectors


async def run(args: argparse.Namespace) -> int:
    plan = load_case_plan(args.plan)
    documents = load_case_context(
        catalog_path=args.catalog,
        document_blueprint_path=args.document_blueprint,
    )
    planner = CaseSeedPlanner(plan, documents)
    supplement_requested = bool(
        args.supplement_split
        or args.supplement_type
        or args.supplement_count
    )
    if supplement_requested:
        if not (
            args.supplement_split
            and args.supplement_type
            and args.supplement_count > 0
        ):
            raise ValueError(
                "定向补充必须同时指定split、type和正数count"
            )
        if args.force:
            raise ValueError("定向补充不能与--force同时使用")
        seeds = planner.build_supplement_seeds(
            split=DatasetSplit(args.supplement_split),
            case_type=EvalCaseType(args.supplement_type),
            count=args.supplement_count,
            round_no=args.supplement_round,
        )
    else:
        seeds = planner.build_pool_seeds()
    seed_by_code = {seed.seed_code: seed for seed in seeds}

    if args.force:
        existing = []
    elif args.raw_output.exists():
        existing = load_generated_case_jsonl(args.raw_output)
    else:
        existing = []
    generated_by_code = {case.case_code: case for case in existing}

    deepseek = DeepSeekClient()
    service = CaseGenerationService(
        client=deepseek,
        prompt_template=args.prompt.read_text(encoding="utf-8"),
    )
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        missing = [
            seed for seed in seeds if seed.seed_code not in generated_by_code
        ]
        batch_size = plan.generation_batch_size
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            cases = await service.generate_batch(
                seeds=batch,
                documents=documents,
            )
            for case in cases:
                errors = validate_generated_case(
                    case=case,
                    seed=seed_by_code[case.case_code],
                    documents=documents,
                )
                if errors:
                    raise ValueError(
                        f"{case.case_code} 确定性校验失败：{errors}"
                    )
                generated_by_code[case.case_code] = case
            write_case_jsonl(
                args.raw_output,
                [
                    generated_by_code[code]
                    for code in sorted(generated_by_code)
                ],
            )
            print(
                json.dumps(
                    {
                        "generated": len(generated_by_code),
                        "requested_seeds": len(seeds),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await deepseek.aclose()

    lexical_unique = []
    seen_questions = set()
    for code in sorted(generated_by_code):
        case = generated_by_code[code]
        normalized = normalize_question(case.question)
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        lexical_unique.append(case)

    embedding = DashScopeEmbeddingClient()
    try:
        vectors = await _embed_questions(
            embedding,
            [case.question for case in lexical_unique],
        )
    finally:
        await embedding.client.aclose()

    decisions = classify_grouped_semantic_duplicates(
        vectors=vectors,
        group_keys=[
            (case.dataset_split.value, case.case_type.value)
            for case in lexical_unique
        ],
    )
    filtered = []
    for index, (case, decision) in enumerate(
        zip(lexical_unique, decisions)
    ):
        if decision.decision == "DUPLICATE":
            continue
        metadata = {
            **case.generation_metadata,
            "semantic_dedup": {
                "decision": decision.decision,
                "similarity": round(decision.similarity, 6),
            },
        }
        if decision.nearest_index is not None:
            nearest = lexical_unique[decision.nearest_index]
            metadata["semantic_dedup"].update(
                {
                    "nearest_case_code": nearest.case_code,
                    "nearest_question": nearest.question,
                    "nearest_expected_action": (
                        nearest.expected_action.value
                    ),
                    "nearest_fact_keys": sorted(
                        {
                            evidence.fact_key
                            for evidence in nearest.evidences
                            if evidence.fact_key
                        }
                    ),
                }
            )
        filtered.append(
            case.model_copy(update={"generation_metadata": metadata})
        )

    write_case_jsonl(args.output, filtered)
    print(
        json.dumps(
            {
                "raw_candidates": len(generated_by_code),
                "lexical_unique": len(lexical_unique),
                "semantic_unique": len(filtered),
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
