import hashlib


class EmbeddingTextBuilder:
    """
    Embedding 输入文本构建器。

    为什么需要它？
    因为直接向量化 chunk.content 有时不够。

    例如业务规则 chunk.content 是：
        已发货订单支持退款，但需扣除物流费用。

    如果加上：
        文档类型：业务规则
        标题路径：售后规则 > 退款规则
        条款：2.3.1

    检索效果通常更好。
    """

    def build_embedding_text(self, chunk: dict) -> str:
        """
        根据 chunk 字典构建 embedding_text。

        chunk 来源：
            MySQL rag_chunk 查询结果。
        """

        chunk_type = chunk.get("chunk_type") or ""
        title = chunk.get("title") or ""
        title_path = chunk.get("title_path") or ""
        content = chunk.get("content") or ""
        keywords = chunk.get("keywords") or ""
        tags = chunk.get("tags") or ""
        source_section = chunk.get("source_section") or ""
        business_domain = chunk.get("business_domain") or ""

        parts: list[str] = []

        parts.append(f"文档类型：{chunk_type}")

        if business_domain:
            parts.append(f"业务域：{business_domain}")

        if title_path:
            parts.append(f"标题路径：{title_path}")
        elif title:
            parts.append(f"标题：{title}")

        if source_section:
            parts.append(f"来源章节：{source_section}")

        if tags:
            parts.append(f"标签：{tags}")

        if keywords:
            parts.append(f"关键词：{keywords}")

        parts.append("正文：")
        parts.append(content)

        return "\n".join(parts).strip()

    def hash_text(self, text: str) -> str:
        """
        计算 embedding_text 的 SHA256。

        作用：
        判断 chunk 内容或 metadata 是否变化。
        如果 hash 没变，就可以跳过重复向量化。
        """

        return hashlib.sha256(text.encode("utf-8")).hexdigest()