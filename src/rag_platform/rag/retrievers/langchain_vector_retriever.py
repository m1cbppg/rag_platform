import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from src.rag_platform.rag.retrieval.vector_retriever import VectorRetriever
from src.rag_platform.rag.retrievers.document_mapper import RetrievalDocumentMapper


class LangChainVectorRetriever(BaseRetriever):
    """
    LangChain Vector Retriever。

    底层使用：
    - 阿里 text-embedding-v4 生成 query embedding；
    - Milvus 做向量检索；
    - MySQL 回查 chunk 内容。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = Field(default=20, description="向量召回数量")
    doc_type: str | None = Field(default=None, description="文档类型过滤")
    business_domain: str | None = Field(default=None, description="业务域过滤")

    _vector_retriever: VectorRetriever = PrivateAttr()
    _mapper: RetrievalDocumentMapper = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._vector_retriever = VectorRetriever()
        self._mapper = RetrievalDocumentMapper()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        异步向量检索。

        为什么是异步？
        因为 VectorRetriever 内部要调用 DashScope API 生成 query embedding。
        """

        hits = await self._vector_retriever.retrieve(
            query=query,
            top_k=self.top_k,
            doc_type=self.doc_type,
            business_domain=self.business_domain,
        )

        return self._mapper.to_documents(hits)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        同步兜底方法。

        注意：
        在 FastAPI 的 async 环境里，不建议调用这个同步方法。
        后续 LangGraph 节点中应该使用 await retriever.ainvoke(query)。
        """

        return asyncio.run(
            self._aget_relevant_documents(
                query=query,
                run_manager=run_manager,
            )
        )