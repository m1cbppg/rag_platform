from abc import ABC, abstractmethod
from typing import Any

from src.rag_platform.domain.chunk import ChunkBuildItem


class BaseChunker(ABC):
    """
    chunker 基类。

    ABC 是抽象基类。
    abstractmethod 表示子类必须实现这个方法。

    chunker 的职责：
    把 structure_json 转成 ChunkBuildItem 列表。
    """

    @abstractmethod
    def build_chunks(
        self,
        structure: dict[str, Any],
        clean_content: str,
    ) -> list[ChunkBuildItem]:
        """
        构建 chunk。

        structure：
            模块 2 保存的结构化解析结果。

        clean_content：
            模块 2 清洗后的文本。
        """
        raise NotImplementedError