import json
from pathlib import Path

from scripts.review_eval_corpus import load_review_documents
from src.rag_platform.evaluation.corpus_models import DocumentBlueprint
from src.rag_platform.evaluation.corpus_services import CorpusFileStore


def test_single_document_review_does_not_require_full_corpus(
    tmp_path: Path,
) -> None:
    selected = DocumentBlueprint(
        source_doc_code="FAQ_SELECTED_001",
        doc_type="FAQ",
        title="已生成文档",
        topic="test",
        version="1.0",
        required_facts=[
            {"fact_key": "fact_1", "description": "事实"},
        ],
        required_identifiers=["F-SELECTED-001"],
        required_sections=["问题"],
    )
    unrelated = selected.model_copy(
        update={
            "source_doc_code": "FAQ_UNRELATED_001",
            "title": "尚未生成文档",
            "required_identifiers": ["F-UNRELATED-001"],
        }
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "FAQ_SELECTED_001.json").write_text(
        json.dumps(
            {
                "source_doc_code": "FAQ_SELECTED_001",
                "title": "已生成文档",
                "doc_type": "FAQ",
                "topic": "test",
                "version": "1.0",
                "sections": [
                    {
                        "section_code": "Q1",
                        "heading": "问题",
                        "content": "F-SELECTED-001 答案。",
                        "aliases": ["怎么处理？", "如何处理？"],
                        "facts": [
                            {"fact_key": "fact_1", "fact_text": "事实"}
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = CorpusFileStore(
        source_dir=source_dir,
        manifest_path=tmp_path / "manifest.jsonl",
    )

    documents = load_review_documents(
        store=store,
        all_blueprints=[selected, unrelated],
        selected_blueprints=[selected],
    )

    assert set(documents) == {"FAQ_SELECTED_001"}
