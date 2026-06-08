from pydantic import BaseModel, Field


class ChunkBuildResponse(BaseModel):
    """
    chunk 构建接口响应。
    """

    doc_id: int = Field(..., description="文档ID")
    chunk_count: int = Field(..., description="生成的chunk数量")
    relation_count: int = Field(..., description="生成的chunk关系数量")
    status: str = Field(..., description="文档状态")
    message: str = Field(..., description="处理结果说明")