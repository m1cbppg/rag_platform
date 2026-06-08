from typing import Any

from src.rag_platform.domain.chunk import ChunkBuildItem, ChunkType
from src.rag_platform.rag.chunkers.base import BaseChunker


class SopChunker(BaseChunker):
    """
    SOP chunker。

    策略：
    1. parent chunk 保存 SOP 总览；
    2. child chunk 按步骤组切分；
    3. 每 3 个步骤一组。
    """

    STEP_GROUP_SIZE = 3

    def build_chunks(
        self,
        structure: dict[str, Any],
        clean_content: str,
    ) -> list[ChunkBuildItem]:
        title = structure.get("title") or "SOP流程"
        scene = structure.get("scene") or ""
        steps = structure.get("steps", [])
        notes = structure.get("notes", [])

        chunks: list[ChunkBuildItem] = []

        parent_key = "sop-parent"

        parent_content = self._build_parent_content(
            title=title,
            scene=scene,
            steps=steps,
            notes=notes,
        )

        chunks.append(
            ChunkBuildItem(
                chunk_type=ChunkType.SOP,
                title=title,
                title_path=f"SOP > {title}",
                content=parent_content,
                summary=f"{title} 总览",
                source_section="SOP总览",
                temp_key=parent_key,
                sort_order=1,
            )
        )

        child_order = 2

        for group_index, step_group in enumerate(
            self._split_steps(steps),
            start=1,
        ):
            group_title = f"{title} - 步骤组 {group_index}"

            content = (
                f"SOP标题：{title}\n"
                f"适用场景：{scene}\n"
                f"步骤组：{group_index}\n"
                f"处理步骤：\n"
                + "\n".join(step_group)
            )

            chunks.append(
                ChunkBuildItem(
                    chunk_type=ChunkType.SOP,
                    title=group_title,
                    title_path=f"SOP > {title} > 步骤组 {group_index}",
                    content=content,
                    summary="\n".join(step_group)[:200],
                    source_section=f"步骤组-{group_index}",
                    parent_temp_key=parent_key,
                    temp_key=f"sop-child-{group_index}",
                    sort_order=child_order,
                )
            )

            child_order += 1

        return chunks

    def _build_parent_content(
        self,
        title: str,
        scene: str,
        steps: list[str],
        notes: list[str],
    ) -> str:
        """
        构建 parent chunk 内容。
        """

        content = f"SOP标题：{title}\n"

        if scene:
            content += f"适用场景：{scene}\n"

        if steps:
            content += "完整处理流程：\n"
            content += "\n".join(steps)
            content += "\n"

        if notes:
            content += "注意事项：\n"
            content += "\n".join(notes)

        return content.strip()

    def _split_steps(self, steps: list[str]) -> list[list[str]]:
        """
        每 STEP_GROUP_SIZE 个步骤切成一组。

        例如 steps = [1,2,3,4,5]
        STEP_GROUP_SIZE = 3

        返回：
        [[1,2,3], [4,5]]
        """

        result: list[list[str]] = []

        for i in range(0, len(steps), self.STEP_GROUP_SIZE):
            result.append(steps[i:i + self.STEP_GROUP_SIZE])

        return result