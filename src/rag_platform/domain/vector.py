from enum import StrEnum


class EmbeddingTaskStatus(StrEnum):
    """
    向量化任务状态。

    PENDING：
        已创建任务，等待向量化。

    PROCESSING：
        正在调用 embedding 模型。

    SUCCESS：
        已成功生成向量并写入 Milvus。

    FAILED：
        向量化或写入 Milvus 失败。

    SKIPPED：
        跳过处理，例如发现 embedding_text_hash 没有变化。
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class VectorCollectionStatus(StrEnum):
    """
    Milvus Collection 状态。

    CREATED：
        已创建 collection。

    LOADED：
        已加载到内存，可以查询。

    FAILED：
        初始化失败。
    """

    CREATED = "CREATED"
    LOADED = "LOADED"
    FAILED = "FAILED"


class EmbeddingOutputType(StrEnum):
    """
    Embedding 输出类型。

    当前项目先使用 dense。
    sparse 和 dense&sparse 后续做混合稀疏向量检索时再扩展。
    """

    DENSE = "dense"
    SPARSE = "sparse"
    DENSE_AND_SPARSE = "dense&sparse"