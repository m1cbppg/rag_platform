from uuid import uuid4

from src.rag_platform.schemas.chat import ChatRequest, ChatResponse


class RagService:
    """
    RAG 应用服务。

    这一层负责对外提供“业务用例”。
    例如：
    - 执行一次 RAG 问答；
    - 保存检索日志；
    - 触发反馈记录。

    模块 1 先返回占位答案。
    后续模块会在这里接入 LangGraph 编排后的完整 RAG 流程。
    """

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        执行一次普通问答。

        async 表示这是异步函数。
        FastAPI 支持异步函数，可以更好地处理网络 IO，例如调用 DeepSeek API。
        """

        trace_id = uuid4().hex

        return ChatResponse(
            answer=(
                "RAG 平台工程骨架已启动。"
                "后续模块会接入文档解析、Milvus 检索、BM25、Rerank 和 DeepSeek 答案生成。"
            ),
            trace_id=trace_id,
        )