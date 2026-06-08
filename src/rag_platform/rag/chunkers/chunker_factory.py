from src.rag_platform.domain.document import DocumentType
from src.rag_platform.rag.chunkers.base import BaseChunker
from src.rag_platform.rag.chunkers.faq_chunker import FaqChunker
from src.rag_platform.rag.chunkers.manual_chunker import ManualChunker
from src.rag_platform.rag.chunkers.rule_chunker import RuleChunker
from src.rag_platform.rag.chunkers.sop_chunker import SopChunker


class ChunkerFactory:
    """
    chunker 工厂。

    根据文档类型返回对应 chunker。
    """

    def get_chunker(self, doc_type: DocumentType) -> BaseChunker:
        if doc_type == DocumentType.FAQ:
            return FaqChunker()

        if doc_type == DocumentType.SOP:
            return SopChunker()

        if doc_type == DocumentType.RULE:
            return RuleChunker()

        if doc_type == DocumentType.MANUAL:
            return ManualChunker()

        raise ValueError(f"不支持的文档类型: {doc_type}")