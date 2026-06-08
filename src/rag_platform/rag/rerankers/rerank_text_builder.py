class RerankTextBuilder:
    """
    Rerank 输入文本构建器。

    Rerank 模型判断的是 query-document 的相关性。
    document 文本最好包含：
    1. 标题路径；
    2. 文档类型；
    3. 来源章节；
    4. 正文内容。

    但不要塞太长，否则会被截断，也会增加成本。
    """

    def build_text(self, document: dict) -> str:
        """
        document 来自 LangGraph State 中的 merged_documents。
        """

        metadata = document.get("metadata") or {}

        title_path = (
            document.get("title_path")
            or metadata.get("title_path")
            or ""
        )
        chunk_type = (
            document.get("chunk_type")
            or metadata.get("chunk_type")
            or ""
        )
        source_section = (
            document.get("source_section")
            or metadata.get("source_section")
            or ""
        )
        content = document.get("page_content") or ""

        parts: list[str] = []

        if chunk_type:
            parts.append(f"文档类型：{chunk_type}")

        if title_path:
            parts.append(f"标题路径：{title_path}")

        if source_section:
            parts.append(f"来源章节：{source_section}")

        parts.append("正文：")
        parts.append(content)

        return "\n".join(parts).strip()