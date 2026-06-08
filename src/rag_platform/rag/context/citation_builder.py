from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.context import Citation, ContextChunk


class CitationBuilder:
    """
    Citation 构建器。

    citation_id 使用 C1、C2、C3 这种形式。
    后续答案生成时可以要求模型按 [C1] 引用。
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def build_citations(
        self,
        chunks: list[ContextChunk],
    ) -> list[Citation]:
        citations: list[Citation] = []

        for index, chunk in enumerate(chunks, start=1):
            citation_id = f"{self.settings.context_citation_prefix}{index}"

            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    title_path=chunk.title_path,
                    source_section=chunk.source_section,
                    chunk_type=chunk.chunk_type,
                    expansion_type=chunk.expansion_type.value,
                    sort_order=index,
                )
            )

        return citations