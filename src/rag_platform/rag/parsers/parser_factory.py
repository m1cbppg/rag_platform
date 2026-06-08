from src.rag_platform.domain.document import DocumentType
from src.rag_platform.rag.parsers.base import BaseDocumentParser
from src.rag_platform.rag.parsers.faq_docx_parser import FaqDocxParser
from src.rag_platform.rag.parsers.manual_pdf_parser import ManualPdfParser
from src.rag_platform.rag.parsers.rule_docx_parser import RuleDocxParser
from src.rag_platform.rag.parsers.sop_pdf_parser import SopPdfParser


class DocumentParserFactory:
    """
    文档解析器工厂。

    工厂模式的作用：
    根据 doc_type 创建对应解析器。

    好处：
    1. 解析器选择逻辑集中管理；
    2. application service 不需要关心具体解析器类；
    3. 后续新增解析器时，只改这里。
    """

    def get_parser(self, doc_type: DocumentType) -> BaseDocumentParser:
        if doc_type == DocumentType.FAQ:
            return FaqDocxParser()

        if doc_type == DocumentType.SOP:
            return SopPdfParser()

        if doc_type == DocumentType.RULE:
            return RuleDocxParser()

        if doc_type == DocumentType.MANUAL:
            return ManualPdfParser()

        raise ValueError(f"不支持的文档类型: {doc_type}")