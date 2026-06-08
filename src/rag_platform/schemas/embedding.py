from pydantic import BaseModel, Field


class EmbeddingTaskCreateResponse(BaseModel):
    """
    创建 embedding 任务响应。
    """

    task_count: int = Field(..., description="创建或更新的任务数量")
    message: str = Field(..., description="处理结果说明")


class EmbeddingTaskRunResponse(BaseModel):
    """
    执行 embedding 任务响应。
    """

    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    message: str = Field(..., description="处理结果说明")