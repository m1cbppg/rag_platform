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
        sub_queries: list[dict] | None = None,
    ) -> tuple[str, str]:
        citation_summary = self._build_citation_summary(citations)
        sub_query_plan = self._build_sub_query_plan(
            sub_queries or []
        )
        sub_query_requirements = (
            "\n".join(
                [
                    "6. 必须逐个回答所有子问题，不能遗漏；",
                    "7. 每个子问题的关键结论必须携带引用；",
                    (
                        "8. 某个子问题证据不足时单独说明，"
                        "不能用其他子问题的证据代替。"
                    ),
                ]
            )
            if sub_queries
            else ""
        )

        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            rewritten_question=rewritten_question or question,
            citation_summary=citation_summary,
            sub_query_plan=sub_query_plan,
            context=context,
            sub_query_requirements=sub_query_requirements,
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

    def _build_sub_query_plan(
        self,
        sub_queries: list[dict],
    ) -> str:
        if not sub_queries:
            return ""
        lines = ["子问题回答计划："]
        for sub_query in sub_queries:
            sub_query_id = sub_query.get("sub_query_id") or ""
            question = sub_query.get("question") or ""
            lines.append(f"- {sub_query_id}：{question}")
        return "\n".join(lines)
