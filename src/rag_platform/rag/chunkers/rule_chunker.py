from typing import Any

from src.rag_platform.domain.chunk import ChunkBuildItem, ChunkType
from src.rag_platform.rag.chunkers.base import BaseChunker


class RuleChunker(BaseChunker):
    """
    业务规则 chunker。

    策略：
    一个条款生成一个 chunk。
    """

    def build_chunks(
        self,
        structure: dict[str, Any],
        clean_content: str,
    ) -> list[ChunkBuildItem]:
        title = structure.get("title") or "业务规则"
        clauses = structure.get("clauses", [])

        chunks: list[ChunkBuildItem] = []

        for index, clause in enumerate(clauses, start=1):
            clause_no = clause.get("clause_no", "").strip()
            title_path = clause.get("title_path", "").strip()
            content_text = clause.get("content", "").strip()
            raw_line = clause.get("raw_line", "").strip()

            if not content_text:
                continue

            source_section = clause_no or f"规则-{index}"

            content = (
                f"文档标题：{title}\n"
                f"章节：{title_path}\n"
                f"条款编号：{clause_no}\n"
                f"规则内容：{content_text}"
            )

            chunks.append(
                ChunkBuildItem(
                    chunk_type=ChunkType.RULE,
                    title=f"{clause_no} {content_text[:30]}",
                    title_path=f"业务规则 > {title} > {title_path}",
                    content=content,
                    summary=content_text[:200],
                    keywords=raw_line,
                    source_section=source_section,
                    temp_key=f"rule-{index}",
                    sort_order=index,
                )
            )

        return chunks