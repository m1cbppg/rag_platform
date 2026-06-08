from pydantic import BaseModel, Field


class VectorCollectionInitResponse(BaseModel):
    """
    Milvus Collection 初始化响应。
    """

    collection_name: str = Field(..., description="Milvus collection名称")
    embedding_model: str = Field(..., description="Embedding模型")
    embedding_dimension: int = Field(..., description="向量维度")
    embedding_output_type: str = Field(..., description="向量输出类型")
    metric_type: str = Field(..., description="向量距离度量")
    index_type: str = Field(..., description="索引类型")
    status: str = Field(..., description="Collection状态")
    message: str = Field(..., description="处理结果说明")