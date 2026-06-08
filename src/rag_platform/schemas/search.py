from pydantic import BaseModel, Field


class SearchIndexInitResponse(BaseModel):
    index_name: str = Field(..., description="ES index 名称")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="说明")


class KeywordIndexTaskCreateResponse(BaseModel):
    task_count: int = Field(..., description="创建或更新任务数量")
    message: str = Field(..., description="说明")


class KeywordIndexTaskRunResponse(BaseModel):
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    message: str = Field(..., description="说明")


class SearchHitResponse(BaseModel):
    chunk_id: int
    score: float
    source: str
    title: str | None = None
    title_path: str | None = None
    content: str | None = None


class SearchTestResponse(BaseModel):
    query: str
    hits: list[SearchHitResponse]