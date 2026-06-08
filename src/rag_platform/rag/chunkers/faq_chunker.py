from typing import Any

from src.rag_platform.domain.chunk import ChunkBuildItem, ChunkType
from src.rag_platform.rag.chunkers.base import BaseChunker


class FaqChunker(BaseChunker):
    """
    FAQ chunker。

    策略：
    一个问答对生成一个 chunk。
    """

    def build_chunks(
        self,
        structure: dict[str, Any],
        clean_content: str,
    ) -> list[ChunkBuildItem]:
        qa_pairs = structure.get("qa_pairs", [])

        chunks: list[ChunkBuildItem] = []

        for index, item in enumerate(qa_pairs, start=1):
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            aliases = item.get("aliases", [])
            tags = item.get("tags", [])

            if not question or not answer:
                continue

            aliases_text = "；".join(aliases)
            tags_text = "；".join(tags)

            content = (
                f"问题：{question}\n"
                f"答案：{answer}"
            )

            if aliases_text:
                content += f"\n同义问法：{aliases_text}"

            if tags_text:
                content += f"\n标签：{tags_text}"

            chunks.append(
                ChunkBuildItem(
                    chunk_type=ChunkType.FAQ,
                    title=question,
                    title_path=f"FAQ > {question}",
                    content=content,
                    summary=answer[:200],
                    keywords=aliases_text,
                    tags=tags_text,
                    source_section=f"FAQ-{index}",
                    temp_key=f"faq-{index}",
                    sort_order=index,
                )
            )

        return chunks