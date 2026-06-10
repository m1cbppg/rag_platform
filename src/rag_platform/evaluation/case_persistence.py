import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.rag_platform.evaluation.models import (
    GeneratedEvalCase,
    MappingStatus,
    ReviewedEvalCase,
)


CaseType = TypeVar("CaseType", bound=BaseModel)


def write_case_jsonl(
    path: Path,
    cases: list[BaseModel],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            _json_line(case.model_dump(mode="json")) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )


def load_generated_case_jsonl(path: Path) -> list[GeneratedEvalCase]:
    return _load_jsonl(path, GeneratedEvalCase)


def load_reviewed_case_jsonl(path: Path) -> list[ReviewedEvalCase]:
    return _load_jsonl(path, ReviewedEvalCase)


def build_frozen_jsonl(cases: list[ReviewedEvalCase]) -> str:
    return "".join(
        _json_line(case.model_dump(mode="json")) + "\n"
        for case in sorted(cases, key=lambda item: item.case_code)
    )


def frozen_content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_required_evidence_mapped(
    cases: list[ReviewedEvalCase],
) -> list[str]:
    errors = []
    for case in cases:
        if any(
            evidence.relevance_grade == 3
            and (
                evidence.mapping_status != MappingStatus.MAPPED
                or evidence.mapped_doc_id is None
                or evidence.mapped_chunk_id is None
            )
            for evidence in case.evidences
        ):
            errors.append(f"{case.case_code}存在未映射的必要证据")
    return errors


def _load_jsonl(path: Path, model_type: type[CaseType]) -> list[CaseType]:
    cases = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            cases.append(model_type.model_validate_json(line))
        except Exception as exc:
            raise ValueError(
                f"{path} 第{line_no}行不是合法评测题"
            ) from exc
    return cases


def _json_line(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
