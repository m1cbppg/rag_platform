from src.rag_platform.rag.answer.answer_prompt import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
)


class AnswerPromptBuilder:
    """
    答案生成 Prompt 构建器。

    输入：
        question
        rewritten_question
        context
        citations

    输出：
        system_prompt
        user_prompt
    """

    def build(
        self,
        question: str,
        rewritten_question: str | None,
        context: str,
        citations: list[dict],
    ) -> tuple[str, str]:
        citation_summary = self._build_citation_summary(citations)

        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            rewritten_question=rewritten_question or question,
            citation_summary=citation_summary,
            context=context,
        )

        return ANSWER_SYSTEM_PROMPT.strip(), user_prompt.strip()

    def _build_citation_summary(self, citations: list[dict]) -> str:
        if not citations:
            return "无可用引用"

        lines: list[str] = []

        for citation in citations:
            citation_id = citation.get("citation_id")
            title_path = citation.get("title_path") or citation.get("title") or "未知标题"
            source_section = citation.get("source_section") or "未知章节"
            chunk_type = citation.get("chunk_type") or "UNKNOWN"

            lines.append(
                f"[{citation_id}] 文档类型：{chunk_type}；标题路径：{title_path}；来源章节：{source_section}"
            )

        return "\n".join(lines)