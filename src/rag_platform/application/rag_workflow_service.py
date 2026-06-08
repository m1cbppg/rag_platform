from uuid import uuid4

from src.rag_platform.domain.rag_state import RagState
from src.rag_platform.rag.graph.rag_retrieval_graph import RagRetrievalGraphBuilder
from src.rag_platform.schemas.rag_workflow import (
    RagRetrievalWorkflowRequest,
    RagRetrievalWorkflowResponse,
    WorkflowDocumentResponse,
)


class RagWorkflowService:
    """
    RAG 工作流服务。

    这一层不直接写 graph 节点逻辑。
    它只负责：
    1. 创建初始 state；
    2. 调用 compiled graph；
    3. 把最终 state 转成 API 响应。
    """

    def __init__(self) -> None:
        builder = RagRetrievalGraphBuilder()
        self.graph = builder.build()

    async def run_retrieval_workflow(
        self,
        request: RagRetrievalWorkflowRequest,
    ) -> RagRetrievalWorkflowResponse:
        """
        执行模块 9 的检索工作流。
        """

        initial_state: RagState = {
            "question": request.question,
            "session_id": request.session_id,
            "business_domain": request.business_domain,
            "trace_id": uuid4().hex,
            "status": "STARTED",
            "error": None,
            "top_k": request.top_k,
        }

        final_state = await self.graph.ainvoke(initial_state)

        merged_documents = [
            self._to_document_response(item)
            for item in final_state.get("merged_documents", [])
        ]

        reranked_documents = [
            self._to_document_response(item)
            for item in final_state.get("reranked_documents", [])
        ]

        return RagRetrievalWorkflowResponse(
            trace_id=final_state.get("trace_id") or initial_state["trace_id"],
            question=request.question,
            rewritten_question=final_state.get("rewritten_question"),
            retrieval_mode=final_state.get("retrieval_mode"),
            target_doc_types=final_state.get("target_doc_types", []),
            status=final_state.get("status", "UNKNOWN"),
            retrieval_quality=final_state.get("retrieval_quality", {}),
            documents=merged_documents,
            rerank_info=final_state.get("rerank_info", {}),
            reranked_documents=reranked_documents,
            # 模块 11 新增
            context=final_state.get("context"),
            citations=final_state.get("citations", []),
            context_build_info=final_state.get("context_build_info", {}),
        )

    def _to_document_response(
            self,
            item: dict,
    ) -> WorkflowDocumentResponse:
        metadata = item.get("metadata") or {}

        return WorkflowDocumentResponse(
            chunk_id=int(item.get("chunk_id")),
            score=item.get("score"),
            source=item.get("source"),
            rerank_score=item.get("rerank_score") or metadata.get("rerank_score"),
            after_rank=item.get("after_rank") or metadata.get("after_rank"),
            title=item.get("title"),
            title_path=item.get("title_path"),
            chunk_type=item.get("chunk_type"),
            business_domain=item.get("business_domain"),
            source_section=item.get("source_section"),
            content=item.get("page_content") or "",
            metadata=metadata,
        )