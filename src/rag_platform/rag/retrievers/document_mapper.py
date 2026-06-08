from langchain_core.documents import Document

from src.rag_platform.domain.search import RetrievalHit


class RetrievalDocumentMapper:
    """
    RetrievalHit -> LangChain Document 转换器。

    为什么要单独抽出来？
    1. BM25 / Vector / Hybrid 都要转 Document；
    2. 统一 metadata 字段，后续 Prompt 引用更稳定；
    3. 避免每个 Retriever 里重复拼 Document。
    """

    def to_document(self, hit: RetrievalHit) -> Document:
        """
        把一个 RetrievalHit 转成 LangChain Document。

        LangChain Document 有两个核心字段：
        1. page_content：正文内容；
        2. metadata：元数据。
        """

        metadata = hit.metadata or {}

        page_content = metadata.get("content") or ""

        doc_metadata = {
            "chunk_id": hit.chunk_id,
            "score": hit.score,
            "source": hit.source,

            "doc_id": metadata.get("doc_id"),
            "chunk_code": metadata.get("chunk_code"),
            "chunk_type": metadata.get("chunk_type"),
            "title": metadata.get("title"),
            "title_path": metadata.get("title_path"),
            "business_domain": metadata.get("business_domain"),
            "version": metadata.get("version"),
            "source_section": metadata.get("source_section"),

            # Hybrid Search 会带这两个分数
            "vector_score": metadata.get("vector_score"),
            "bm25_score": metadata.get("bm25_score"),
        }

        return Document(
            page_content=page_content,
            metadata=doc_metadata,
        )

    def to_documents(self, hits: list[RetrievalHit]) -> list[Document]:
        """
        批量转换。
        """

        return [self.to_document(hit) for hit in hits]