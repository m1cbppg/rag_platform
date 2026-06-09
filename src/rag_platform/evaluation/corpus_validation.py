import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    GeneratedSourceDocument,
)
from src.rag_platform.evaluation.models import SourceDocumentType


EXPECTED_DOCUMENT_COUNTS = {
    SourceDocumentType.FAQ: 12,
    SourceDocumentType.SOP: 10,
    SourceDocumentType.RULE: 12,
    SourceDocumentType.MANUAL: 6,
}


@dataclass
class BlueprintValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def load_document_blueprints(path: Path) -> list[DocumentBlueprint]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("文档蓝图文件的顶层结构必须是数组")
    return [DocumentBlueprint.model_validate(item) for item in payload]


def validate_blueprint_plan(
    blueprints: list[DocumentBlueprint],
) -> BlueprintValidationReport:
    errors: list[str] = []
    codes = [item.source_doc_code for item in blueprints]

    if len(blueprints) != 40:
        errors.append(f"文档蓝图必须正好包含40条，实际为{len(blueprints)}条")
    if len(codes) != len(set(codes)):
        errors.append("source_doc_code 存在重复")

    counts = Counter(item.doc_type for item in blueprints)
    for doc_type, expected in EXPECTED_DOCUMENT_COUNTS.items():
        actual = counts.get(doc_type, 0)
        if actual != expected:
            errors.append(
                f"{doc_type.value} 文档数量应为{expected}，实际为{actual}"
            )

    known_codes = set(codes)
    for item in blueprints:
        references = [*item.conflicts_with]
        if item.supersedes:
            references.append(item.supersedes)
        for reference in references:
            if reference not in known_codes:
                errors.append(
                    f"{item.source_doc_code} 引用了不存在的文档 {reference}"
                )

    rule_groups = Counter(
        item.version_group
        for item in blueprints
        if item.doc_type == SourceDocumentType.RULE and item.version_group
    )
    paired_groups = [group for group, count in rule_groups.items() if count >= 2]
    if len(paired_groups) < 4:
        errors.append("RULE 至少需要四个包含新旧版本的 version_group")

    return BlueprintValidationReport(errors=errors)


def validate_generated_document(
    blueprint: DocumentBlueprint,
    document: GeneratedSourceDocument,
) -> list[str]:
    errors: list[str] = []
    comparable_fields = (
        "source_doc_code",
        "title",
        "doc_type",
        "topic",
        "version",
        "effective_from",
        "effective_to",
    )
    for field_name in comparable_fields:
        expected = getattr(blueprint, field_name)
        actual = getattr(document, field_name)
        if expected != actual:
            errors.append(
                f"{field_name} 与蓝图不一致：expected={expected}, actual={actual}"
            )

    actual_fact_keys = document.fact_keys()
    for required_fact in blueprint.required_facts:
        if required_fact.fact_key not in actual_fact_keys:
            errors.append(f"缺少必要事实 fact_key={required_fact.fact_key}")

    text = document.plain_text()
    for identifier in blueprint.required_identifiers:
        if identifier not in text:
            errors.append(f"缺少必要标识符 {identifier}")

    normalized_text = _normalize_text(text)
    for section in blueprint.required_sections:
        if _normalize_text(section) not in normalized_text:
            errors.append(f"缺少必要章节或结构 {section}")

    errors.extend(_validate_type_contract(document))
    return errors


def _validate_type_contract(document: GeneratedSourceDocument) -> list[str]:
    errors: list[str] = []
    text = document.plain_text()

    if document.doc_type == SourceDocumentType.FAQ:
        if not 8 <= len(document.sections) <= 15:
            errors.append("FAQ 必须包含8到15组问答")
        for section in document.sections:
            if not 2 <= len(section.aliases) <= 4:
                errors.append(
                    f"FAQ章节 {section.section_code} 必须包含2到4个同义问法"
                )

    if document.doc_type == SourceDocumentType.SOP:
        step_count = len(re.findall(r"(?m)^\s*\d+[.、]\s*", text))
        if step_count < 4:
            errors.append("SOP 至少需要4个编号步骤")
        for marker in ("适用场景", "前置检查", "异常", "升级"):
            if marker not in text:
                errors.append(f"SOP 缺少{marker}")

    if document.doc_type == SourceDocumentType.RULE:
        for marker in ("适用", "例外", "优先级"):
            if marker not in text:
                errors.append(f"RULE 缺少{marker}说明")

    if document.doc_type == SourceDocumentType.MANUAL:
        step_count = len(re.findall(r"(?m)^\s*\d+[.、]\s*", text))
        if step_count < 3:
            errors.append("MANUAL 至少需要3个编号步骤")
        for marker in ("菜单路径", "字段", "【", "错误"):
            if marker not in text:
                errors.append(f"MANUAL 缺少{marker}信息")

    return errors


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()
