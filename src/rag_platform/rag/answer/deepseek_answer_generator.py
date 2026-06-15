from collections.abc import AsyncGenerator

from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.deepseek import DeepSeekClient
from src.rag_platform.rag.answer.answer_prompt_builder import AnswerPromptBuilder


class DeepSeekAnswerGenerator:
    """
    DeepSeek 答案生成器。

    只负责：
    1. 构建 prompt；
    2. 调用 DeepSeek；
    3. 返回答案文本或流式 delta。

    不负责：
    1. 检索；
    2. rerank；
    3. context 构建；
    4. 日志落库。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = DeepSeekClient()
        self.prompt_builder = AnswerPromptBuilder()

    async def generate(
        self,
        question: str,
        rewritten_question: str | None,
        context: str,
        citations: list[dict],
        sub_queries: list[dict] | None = None,
    ) -> str:
        system_prompt, user_prompt = self.prompt_builder.build(
            question=question,
            rewritten_question=rewritten_question,
            context=context,
            citations=citations,
            sub_queries=sub_queries,
        )

        return await self.client.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.settings.answer_model,
            temperature=self.settings.answer_temperature,
            max_tokens=self.settings.answer_max_tokens,
        )

    async def stream_generate(
        self,
        question: str,
        rewritten_question: str | None,
        context: str,
        citations: list[dict],
        sub_queries: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        system_prompt, user_prompt = self.prompt_builder.build(
            question=question,
            rewritten_question=rewritten_question,
            context=context,
            citations=citations,
            sub_queries=sub_queries,
        )

        async for delta in self.client.stream_chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.settings.answer_model,
            temperature=self.settings.answer_temperature,
            max_tokens=self.settings.answer_max_tokens,
        ):
            yield delta
