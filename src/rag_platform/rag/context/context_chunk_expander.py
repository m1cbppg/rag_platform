from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.chunk import ChunkRelationType
from src.rag_platform.domain.context import ContextChunk, ContextExpansionType
from src.rag_platform.infrastructure.repositories.context_repository import ContextRepository


class ContextChunkExpander:
    """
    Context chunk 扩展器。

    根据不同文档类型和 chunk_relation 扩展上下文。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.repository = ContextRepository()

    def expand(self, base_chunks: list[ContextChunk]) -> list[ContextChunk]:
        """
        对基础 chunk 做关系扩展。

        返回：
            base chunks + expanded chunks
        """

        expanded_chunks: list[ContextChunk] = []
        expanded_chunks.extend(base_chunks)

        for base_chunk in base_chunks:
            related_ids_with_type = self._find_related_ids(base_chunk)

            if not related_ids_with_type:
                continue

            related_ids = [
                item[0]
                for item in related_ids_with_type
            ]

            related_map = self.repository.get_chunks_by_ids(related_ids)

            relation_type_map = {
                chunk_id: relation_type
                for chunk_id, relation_type in related_ids_with_type
            }

            count = 0

            for related_id in related_ids:
                if count >= self.settings.context_max_expanded_chunks_per_hit:
                    break

                row = related_map.get(related_id)
                if row is None:
                    continue

                relation_type = relation_type_map.get(related_id)

                expanded_chunks.append(
                    self._row_to_context_chunk(
                        row=row,
                        base_chunk=base_chunk,
                        relation_type=relation_type,
                    )
                )
                count += 1

        return expanded_chunks

    def _find_related_ids(
        self,
        chunk: ContextChunk,
    ) -> list[tuple[int, str]]:
        """
        根据 chunk 类型选择扩展关系。
        """

        relation_types: list[str] = []

        chunk_type = (chunk.chunk_type or "").upper()

        if chunk_type in ["SOP", "MANUAL"]:
            if self.settings.context_expand_parent:
                relation_types.append(ChunkRelationType.PARENT_CHILD.value)

            if self.settings.context_expand_previous_next:
                relation_types.append(ChunkRelationType.PREVIOUS_NEXT.value)

        elif chunk_type == "RULE":
            if self.settings.context_expand_previous_next:
                relation_types.append(ChunkRelationType.PREVIOUS_NEXT.value)

            if self.settings.context_expand_same_section:
                relation_types.append(ChunkRelationType.SAME_SECTION.value)

        else:
            # FAQ 默认不扩展，保持精准。
            relation_types = []

        return self.repository.get_related_chunk_ids(
            chunk_id=chunk.chunk_id,
            relation_types=relation_types,
            limit=self.settings.context_max_expanded_chunks_per_hit,
        )

    def _row_to_context_chunk(
        self,
        row: dict,
        base_chunk: ContextChunk,
        relation_type: str | None,
    ) -> ContextChunk:
        """
        数据库 row 转 ContextChunk。
        """

        expansion_type = self._relation_to_expansion_type(relation_type)

        # 扩展 chunk 的排序分数略低于原始命中 chunk。
        sort_score = base_chunk.sort_score - 0.01

        return ContextChunk(
            chunk_id=int(row["id"]),
            doc_id=row.get("doc_id"),
            content=row.get("content") or "",
            title=row.get("title"),
            title_path=row.get("title_path"),
            chunk_type=row.get("chunk_type"),
            business_domain=row.get("business_domain"),
            source_section=row.get("source_section"),
            score=base_chunk.score,
            rerank_score=base_chunk.rerank_score,
            source=base_chunk.source,
            expansion_type=expansion_type,
            original_rank=base_chunk.original_rank,
            sort_score=sort_score,
            metadata={
                **(base_chunk.metadata or {}),
                "expanded_from_chunk_id": base_chunk.chunk_id,
                "relation_type": relation_type,
            },
        )

    def _relation_to_expansion_type(
        self,
        relation_type: str | None,
    ) -> ContextExpansionType:
        if relation_type == ChunkRelationType.PARENT_CHILD.value:
            return ContextExpansionType.PARENT

        if relation_type == ChunkRelationType.PREVIOUS_NEXT.value:
            return ContextExpansionType.PREVIOUS_NEXT

        if relation_type == ChunkRelationType.SAME_SECTION.value:
            return ContextExpansionType.SAME_SECTION

        return ContextExpansionType.SELF
