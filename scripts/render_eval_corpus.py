import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.core.config import get_settings  # noqa: E402
from src.rag_platform.evaluation.corpus_models import (  # noqa: E402
    ReviewedDocumentOutcome,
)
from src.rag_platform.evaluation.corpus_renderer import CorpusRenderer  # noqa: E402
from src.rag_platform.evaluation.corpus_validation import (  # noqa: E402
    load_document_blueprints,
    validate_blueprint_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染并回读验证M2源文档")
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path("evaluation/blueprints/ecommerce_document_plan.json"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("evaluation/corpus/reviews"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/corpus/rendered"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/corpus/render_manifest.jsonl"),
    )
    parser.add_argument("--font-path")
    parser.add_argument("--code", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    blueprints = load_document_blueprints(args.blueprint)
    report = validate_blueprint_plan(blueprints)
    if not report.is_valid:
        raise ValueError("蓝图校验失败：" + "；".join(report.errors))

    selected_codes = set(args.code)
    selected = [
        item
        for item in blueprints
        if not selected_codes or item.source_doc_code in selected_codes
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    settings = get_settings()
    font_path = args.font_path or settings.eval_pdf_font_path or None
    renderer = CorpusRenderer(pdf_font_path=font_path)
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rendered = 0
    skipped = 0

    with args.manifest.open("a", encoding="utf-8") as manifest:
        for blueprint in selected:
            review_path = args.reviews / f"{blueprint.source_doc_code}.json"
            if not review_path.exists():
                raise FileNotFoundError(
                    f"缺少审核结果：{review_path}"
                )
            outcome = ReviewedDocumentOutcome.model_validate_json(
                review_path.read_text(encoding="utf-8")
            )
            if not outcome.review.passed:
                raise ValueError(
                    f"{blueprint.source_doc_code} 未通过审核，禁止渲染"
                )

            expected_suffix = (
                ".docx"
                if blueprint.doc_type.value in {"FAQ", "RULE"}
                else ".pdf"
            )
            output_path = args.output / (
                blueprint.source_doc_code + expected_suffix
            )
            if output_path.exists() and not args.force:
                skipped += 1
                print(f"SKIP {blueprint.source_doc_code}")
                continue

            output_path = renderer.render(outcome.document, args.output)
            verification = renderer.verify(
                blueprint,
                outcome.document,
                output_path,
            )
            if not verification.is_valid:
                output_path.unlink(missing_ok=True)
                raise ValueError(
                    f"{blueprint.source_doc_code} 渲染回读失败："
                    + "；".join(verification.errors)
                )

            entry = {
                "source_doc_code": blueprint.source_doc_code,
                "path": output_path.as_posix(),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "extracted_text_length": len(verification.extracted_text),
                "status": "RENDERED_AND_VERIFIED",
            }
            manifest.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            manifest.flush()
            rendered += 1
            print(f"RENDERED {blueprint.source_doc_code}")

    print(f"完成：渲染{rendered}篇，跳过{skipped}篇")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
