from langchain_core.documents import Document


class GraphDocumentCodec:
    """
    LangGraph 工作流中的 Document 编解码器。

    为什么需要它？
    1. LangChain Retriever 返回 Document；
    2. LangGraph State 最好保存可 JSON 序列化的数据；
    3. 后续日志、SSE、数据库落库都更方便。
    """

    def document_to_dict(self, document: Document) -> dict:
        """
        LangChain Document 转普通 dict。
        """

        metadata = document.metadata or {}

        return {
            "page_content": document.page_content,
            "metadata": metadata,
            "chunk_id": metadata.get("chunk_id"),
            "score": metadata.get("score"),
            "source": metadata.get("source"),
            "title": metadata.get("title"),
            "title_path": metadata.get("title_path"),
            "chunk_type": metadata.get("chunk_type"),
            "business_domain": metadata.get("business_domain"),
            "source_section": metadata.get("source_section"),
        }

    def documents_to_dicts(self, documents: list[Document]) -> list[dict]:
        """
        批量转换 Document。
        """

        return [
            self.document_to_dict(document)
            for document in documents
        ]

    def dict_to_document(self, item: dict) -> Document:
        """
        普通 dict 转回 LangChain Document。
        """

        return Document(
            page_content=item.get("page_content") or "",
            metadata=item.get("metadata") or {},
        )