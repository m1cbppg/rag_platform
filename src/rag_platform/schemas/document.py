from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """
    文档上传并解析后的响应模型。

    BaseModel 来自 Pydantic。
    FastAPI 会根据它自动生成接口文档，也会校验返回字段。
    """

    doc_id: int = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    doc_type: str = Field(..., description="文档类型")
    status: str = Field(..., description="文档状态")
    message: str = Field(..., description="处理结果说明")