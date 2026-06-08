from typing import Any

from src.rag_platform.domain.chunk import ChunkBuildItem, ChunkType
from src.rag_platform.rag.chunkers.base import BaseChunker


class ManualChunker(BaseChunker):
    """
    操作手册 chunker。

    当前策略：
    把一个操作任务生成一个 chunk。

    如果后续操作手册结构更规范，可以按 title_path 拆成多个操作任务 chunk。
    """

    def build_chunks(
        self,
        structure: dict[str, Any],
        clean_content: str,
    ) -> list[ChunkBuildItem]:
        title = structure.get("title") or "操作手册"
        title_path = structure.get("title_path") or title
        steps = structure.get("steps", [])
        button_names = structure.get("button_names", [])

        if not steps and not clean_content:
            return []

        buttons_text = "；".join(button_names)

        content = (
            f"操作手册：{title}\n"
            f"标题路径：{title_path}\n"
        )

        if steps:
            content += "操作步骤：\n"
            content += "\n".join(steps)
        else:
            content += clean_content

        if buttons_text:
            content += f"\n涉及按钮：{buttons_text}"

        return [
            ChunkBuildItem(
                chunk_type=ChunkType.MANUAL,
                title=title,
                title_path=f"操作手册 > {title_path}",
                content=content,
                summary="\n".join(steps)[:200] if steps else clean_content[:200],
                keywords=buttons_text,
                source_section=title_path,
                temp_key="manual-1",
                sort_order=1,
            )
        ]