from typing import Any

from pydantic import BaseModel, Field


class RerankTestRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    documents: list[dict[str, Any]] = Field(..., description="候选文档")


class RerankTestResponse(BaseModel):
    trace_id: str
    rerank_info: dict[str, Any]
    documents: list[dict[str, Any]]