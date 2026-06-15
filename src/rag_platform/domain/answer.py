from enum import StrEnum


class AnswerStatus(StrEnum):
    """
    答案生成状态。
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REFUSED = "REFUSED"
    CLARIFIED = "CLARIFIED"
    STREAMING = "STREAMING"


class ChatStreamEventType(StrEnum):
    """
    SSE 事件类型。
    """

    TRACE = "trace"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    DELTA = "delta"
    DONE = "done"
    ERROR = "error"
