from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.context import (
    ContextBuildResult,
    ContextBuildStatus,
    ContextChunk,
    ContextExpansionType,
)
from src.rag_platform.rag.context.citation_builder import CitationBuilder
from src.rag_platform.rag.context.context_chunk_expander import ContextChunkExpander
from src.rag_platform.rag.context.token_estimator import TokenEstimator


class ContextBuilder:
    """
    上下文构建器。

    职责：
    1. 从 reranked_documents 中提取基础 chunk；
    2. 做 parent / previous-next 扩展；
    3. 去重；
    4. 按排序分数排序；
    5. 控制 token 预算；
    6. 构建 context 文本；
    7. 构建 citations。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.expander = ContextChunkExpander()
        self.token_estimator = TokenEstimator()
        self.citation_builder = CitationBuilder()

    def build(
        self,
        documents: list[dict],
    ) -> ContextBuildResult:
        """
        构建上下文。
        """

        if not documents:
            return ContextBuildResult(
                context="",
                citations=[],
                used_chunks=[],
                estimated_tokens=0,
                status=ContextBuildStatus.EMPTY,
                message="没有可用于构建上下文的候选文档",
            )

        base_chunks = self._documents_to_context_chunks(documents)

        expanded_chunks = self.expander.expand(base_chunks)

        deduped_chunks = self._deduplicate_chunks(expanded_chunks)

        sorted_chunks = sorted(
            deduped_chunks,
            key=lambda item: item.sort_score,
            reverse=True,
        )

        selected_chunks, estimated_tokens, truncated = self._select_by_budget(sorted_chunks)

        citations = self.citation_builder.build_citations(selected_chunks)

        context = self._render_context(
            chunks=selected_chunks,
            citations=citations,
        )

        status = (
            ContextBuildStatus.TRUNCATED
            if truncated
            else ContextBuildStatus.SUCCESS
        )

        return ContextBuildResult(
            context=context,
            citations=citations,
            used_chunks=selected_chunks,
            estimated_tokens=estimated_tokens,
            status=status,
            message="上下文构建完成",
        )

    def _documents_to_context_chunks(
        self,
        documents: list[dict],
    ) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []

        for index, document in enumerate(documents, start=1):
            metadata = document.get("metadata") or {}

            chunk_id = document.get("chunk_id") or metadata.get("chunk_id")
            if chunk_id is None:
                continue

            score = document.get("score")
            rerank_score = document.get("rerank_score") or metadata.get("rerank_score")

            # sort_score 优先用 rerank_score，其次用 score。
            sort_score = float(rerank_score or score or 0.0)

            chunks.append(
                ContextChunk(
                    chunk_id=int(chunk_id),
                    doc_id=metadata.get("doc_id"),
                    content=document.get("page_content") or "",
                    title=document.get("title") or metadata.get("title"),
                    title_path=document.get("title_path") or metadata.get("title_path"),
                    chunk_type=document.get("chunk_type") or metadata.get("chunk_type"),
                    business_domain=document.get("business_domain") or metadata.get("business_domain"),
                    source_section=document.get("source_section") or metadata.get("source_section"),
                    score=score,
                    rerank_score=rerank_score,
                    source=document.get("source") or metadata.get("source"),
                    expansion_type=ContextExpansionType.SELF,
                    original_rank=index,
                    sort_score=sort_score,
                    metadata=metadata,
                )
            )

        return chunks

    def _deduplicate_chunks(
        self,
        chunks: list[ContextChunk],
    ) -> list[ContextChunk]:
        """
        按 chunk_id 去重。

        同一个 chunk 既可能是原始命中，也可能是扩展命中。
        规则：
        1. SELF 优先级最高；
        2. sort_score 更高者优先。
        """

        result: dict[int, ContextChunk] = {}

        for chunk in chunks:
            existed = result.get(chunk.chunk_id)

            if existed is None:
                result[chunk.chunk_id] = chunk
                continue

            if existed.expansion_type != ContextExpansionType.SELF and chunk.expansion_type == ContextExpansionType.SELF:
                result[chunk.chunk_id] = chunk
                continue

            if chunk.sort_score > existed.sort_score:
                result[chunk.chunk_id] = chunk

        return list(result.values())

    def _select_by_budget(
        self,
        chunks: list[ContextChunk],
    ) -> tuple[list[ContextChunk], int, bool]:
        """
        根据 token 预算选择 chunk。
        """

        selected: list[ContextChunk] = []
        total_tokens = 0
        truncated = False

        for chunk in chunks:
            if len(selected) >= self.settings.context_max_chunks:
                truncated = True
                break

            chunk_text = self._render_single_chunk_preview(chunk)
            chunk_tokens = self.token_estimator.estimate(chunk_text)

            if total_tokens + chunk_tokens > self.settings.context_max_tokens:
                truncated = True
                break

            selected.append(chunk)
            total_tokens += chunk_tokens

        return selected, total_tokens, truncated

    def _render_context(
        self,
        chunks: list[ContextChunk],
        citations,
    ) -> str:
        """
        渲染最终 context 文本。
        """

        blocks: list[str] = []

        for chunk, citation in zip(chunks, citations):
            header = f"[{citation.citation_id}]"

            if self.settings.context_include_metadata_header:
                title_path = chunk.title_path or chunk.title or "未知标题"
                section = chunk.source_section or "未知章节"
                chunk_type = chunk.chunk_type or "UNKNOWN"

                header += f" 文档类型：{chunk_type}；标题路径：{title_path}；来源：{section}"

            block = f"{header}\n{chunk.content}".strip()
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    def _render_single_chunk_preview(
        self,
        chunk: ContextChunk,
    ) -> str:
        """
        用于估算 token 的 chunk 文本。
        """

        title_path = chunk.title_path or chunk.title or ""
        source_section = chunk.source_section or ""

        return f"{title_path}\n{source_section}\n{chunk.content}"