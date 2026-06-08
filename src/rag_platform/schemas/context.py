from typing import Any

from pydantic import BaseModel, Field


class ContextBuildTestRequest(BaseModel):
    trace_id: str | None = Field(default=None, description="追踪ID")
    query: str = Field(..., description="用户问题")
    documents: list[dict[str, Any]] = Field(..., description="候选文档")


class ContextBuildTestResponse(BaseModel):
    trace_id: str
    context: str
    citations: list[dict[str, Any]]
    context_build_info: dict[str, Any]