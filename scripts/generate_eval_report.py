import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.evaluation.baseline_report import (  # noqa: E402
    build_baseline_report,
    render_markdown,
)
from src.rag_platform.evaluation.dataset_repository import (  # noqa: E402
    DatasetRepository,
)


_RUN_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据已落库的RAG评测运行生成中文基线报告",
    )
    parser.add_argument("--run-code", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/reports"),
    )
    return parser.parse_args()


def validate_run_code(value: str) -> str:
    normalized = value.strip()
    if not _RUN_CODE_PATTERN.fullmatch(normalized):
        raise ValueError(
            "run_code只能包含字母、数字、下划线、点和连字符，长度不超过64"
        )
    return normalized


def generate_report(
    *,
    repository: DatasetRepository,
    run_code: str,
    output_dir: Path,
) -> dict[str, Any]:
    normalized_run_code = validate_run_code(run_code)
    run = repository.find_run_by_code(normalized_run_code)
    if run is None:
        raise ValueError(f"评测运行不存在：{normalized_run_code}")
    run_id = int(run["id"])
    case_results = repository.list_run_case_results(run_id)
    hits = repository.list_run_retrieval_hits(run_id)
    evidences = repository.list_run_evidences(run_id)
    system_diagnostics = repository.get_run_domain_diagnostics(run_id)
    report = build_baseline_report(
        run=run,
        case_results=case_results,
        hits=hits,
        evidences=evidences,
        system_diagnostics=system_diagnostics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{normalized_run_code}.json"
    markdown_path = output_dir / f"{normalized_run_code}.md"
    _atomic_write(
        json_path,
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_write(markdown_path, render_markdown(report))
    return {
        "run_id": run_id,
        "run_code": normalized_run_code,
        "json_path": json_path,
        "markdown_path": markdown_path,
        "overview": report["overview"],
        "attribution_summary": report["attribution_summary"],
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    result = generate_report(
        repository=DatasetRepository(),
        run_code=args.run_code,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                **result,
                "json_path": str(result["json_path"]),
                "markdown_path": str(result["markdown_path"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
