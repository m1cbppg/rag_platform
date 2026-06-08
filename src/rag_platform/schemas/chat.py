from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    普通问答请求体。

    BaseModel 来自 Pydantic。
    它可以帮我们做：
    1. 字段类型校验；
    2. 默认值处理；
    3. 自动生成接口文档。
    """

    question: str = Field(..., description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID")
    business_domain: str | None = Field(default=None, description="业务域，例如：订单、支付、用户、报表、客服流程")


class ChatResponse(BaseModel):
    """
    普通问答响应体。
    模块 1 先返回占位结果。
    后续模块会接入真正的 LangGraph RAG 流程。
    """

    answer: str = Field(..., description="回答内容")
    trace_id: str = Field(..., description="本次请求的追踪 ID")