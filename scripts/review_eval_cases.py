import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.case_persistence import (  # noqa: E402
    load_generated_case_jsonl,
    load_reviewed_case_jsonl,
    write_case_jsonl,
)
from src.rag_platform.evaluation.case_planner import (  # noqa: E402
    load_case_context,
    load_case_plan,
)
from src.rag_platform.evaluation.case_services import (  # noqa: E402
    CaseReviewService,
    retain_reviews_for_recheck,
    select_reviewed_cases,
)
from src.rag_platform.evaluation.models import EvalCaseType  # noqa: E402
from src.rag_platform.infrastructure.dashscope_chat import (  # noqa: E402
    DashScopeChatClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用百炼Qwen审核M4评测题并选出精确300题",
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
        default=Path("evaluation/prompts/case_review.txt"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.generated.jsonl"),
    )
    parser.add_argument(
        "--all-reviews",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.reviewed.all.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/datasets/rag_eval_v1.reviewed.jsonl"),
    )
    parser.add_argument("--recheck-rejected", action="store_true")
    parser.add_argument(
        "--recheck-type",
        choices=[item.value for item in EvalCaseType],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    plan = load_case_plan(args.plan)
    documents = load_case_context(
        catalog_path=args.catalog,
        document_blueprint_path=args.document_blueprint,
    )
    generated = load_generated_case_jsonl(args.input)
    if args.force or not args.all_reviews.exists():
        existing = []
    else:
        existing = load_reviewed_case_jsonl(args.all_reviews)
    existing = retain_reviews_for_recheck(
        existing,
        recheck_rejected=args.recheck_rejected,
        case_type=(
            EvalCaseType(args.recheck_type)
            if args.recheck_type
            else None
        ),
    )
    reviewed_by_code = {case.case_code: case for case in existing}
    missing = [
        case for case in generated if case.case_code not in reviewed_by_code
    ]

    qwen = DashScopeChatClient()
    service = CaseReviewService(
        reviewer=qwen,
        prompt_template=args.prompt.read_text(encoding="utf-8"),
    )
    try:
        for start in range(0, len(missing), plan.generation_batch_size):
            batch = missing[start:start + plan.generation_batch_size]
            reviewed = await service.review_batch(
                cases=batch,
                documents=documents,
            )
            for case in reviewed:
                reviewed_by_code[case.case_code] = case
            write_case_jsonl(
                args.all_reviews,
                [
                    reviewed_by_code[code]
                    for code in sorted(reviewed_by_code)
                ],
            )
            print(
                json.dumps(
                    {
                        "reviewed": len(reviewed_by_code),
                        "candidate_count": len(generated),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await qwen.aclose()

    selected = select_reviewed_cases(
        list(reviewed_by_code.values()),
        plan,
    )
    write_case_jsonl(args.output, selected)
    print(
        json.dumps(
            {
                "reviewed_candidates": len(reviewed_by_code),
                "passed_candidates": sum(
                    case.review_status.value == "PASSED"
                    for case in reviewed_by_code.values()
                ),
                "selected_cases": len(selected),
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
