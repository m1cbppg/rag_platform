from pathlib import Path

import pytest

from src.rag_platform.evaluation.corpus_models import (
    DocumentBlueprint,
    GeneratedSourceDocument,
)
from src.rag_platform.evaluation.corpus_renderer import CorpusRenderer


def _blueprint(doc_type: str, code: str, identifier: str) -> DocumentBlueprint:
    return DocumentBlueprint(
        source_doc_code=code,
        doc_type=doc_type,
        title=f"{doc_type} 测试文档",
        topic="test",
        version="1.0",
        effective_from="2026-01-01" if doc_type == "RULE" else None,
        required_facts=[
            {"fact_key": "required_fact", "description": "必须保留的事实"},
        ],
        required_identifiers=[identifier],
        required_sections=["基础信息"],
    )


def _document(doc_type: str, code: str, identifier: str) -> GeneratedSourceDocument:
    aliases = ["怎么处理？", "这个问题怎么办？"] if doc_type == "FAQ" else []
    content_by_type = {
        "FAQ": f"{identifier} 对应的处理时限为两个工作日。",
        "RULE": f"{identifier} 适用于当前业务。例外条件为人工复核，优先级为高。",
        "SOP": (
            f"适用场景：{identifier} 客诉。\n"
            "前置检查：核对订单状态。\n"
            "1. 打开售后工单。\n"
            "2. 核对用户凭证。\n"
            "3. 提交处理结果。\n"
            "异常分支：凭证缺失时暂停。\n"
            "升级条件：连续两次失败时升级。"
        ),
        "MANUAL": (
            f"菜单路径：售后管理 > 工单中心。\n"
            f"错误提示：{identifier}。\n"
            "1. 点击【查询】按钮。\n"
            "2. 填写订单编号字段。\n"
            "3. 点击【提交】按钮。"
        ),
    }
    return GeneratedSourceDocument.model_validate(
        {
            "source_doc_code": code,
            "title": f"{doc_type} 测试文档",
            "doc_type": doc_type,
            "topic": "test",
            "version": "1.0",
            "sections": [
                {
                    "section_code": "S1",
                    "heading": "基础信息",
                    "content": content_by_type[doc_type],
                    "aliases": aliases,
                    "facts": [
                        {
                            "fact_key": "required_fact",
                            "fact_text": "必须保留的事实",
                        }
                    ],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ("doc_type", "code", "identifier", "suffix"),
    [
        ("FAQ", "FAQ_TEST_001", "F-TEST-001", ".docx"),
        ("RULE", "RULE_TEST_001", "R-TEST-001", ".docx"),
        ("SOP", "SOP_TEST_001", "S-TEST-001", ".pdf"),
        ("MANUAL", "MANUAL_TEST_001", "E-TEST-001", ".pdf"),
    ],
)
def test_rendered_document_can_be_reparsed_without_identifier_loss(
    tmp_path: Path,
    doc_type: str,
    code: str,
    identifier: str,
    suffix: str,
) -> None:
    blueprint = _blueprint(doc_type, code, identifier)
    document = _document(doc_type, code, identifier)
    renderer = CorpusRenderer()

    output_path = renderer.render(document, tmp_path)
    verification = renderer.verify(blueprint, document, output_path)

    assert output_path.suffix == suffix
    assert verification.is_valid, verification.errors
    assert identifier in verification.extracted_text
