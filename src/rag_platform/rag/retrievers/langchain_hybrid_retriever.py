import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from src.rag_platform.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag_platform.rag.retrieval.hybrid_fusion import HybridFusion
from src.rag_platform.rag.retrieval.vector_retriever import VectorRetriever
from src.rag_platform.rag.retrievers.document_mapper import RetrievalDocumentMapper


class LangChainHybridRetriever(BaseRetriever):
    """
    LangChain Hybrid Retriever。

    Hybrid Search 不是新的存储引擎，而是召回策略：
    1. ES BM25 召回；
    2. Milvus Vector 召回；
    3. 分数融合；
    4. 返回统一 Document。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = Field(default=20, description="最终返回数量")
    vector_top_k: int = Field(default=20, description="向量召回数量")
    bm25_top_k: int = Field(default=20, description="BM25召回数量")
    doc_type: str | None = Field(default=None, description="文档类型过滤")
    business_domain: str | None = Field(default=None, description="业务域过滤")

    _bm25_retriever: BM25Retriever = PrivateAttr()
    _vector_retriever: VectorRetriever = PrivateAttr()
    _fusion: HybridFusion = PrivateAttr()
    _mapper: RetrievalDocumentMapper = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """
        初始化底层检索器。

        只有创建 HybridRetriever 时，才会初始化 VectorRetriever。
        这样 BM25 单独使用时不会依赖 Milvus / DashScope。
        """

        self._bm25_retriever = BM25Retriever()
        self._vector_retriever = VectorRetriever()
        self._fusion = HybridFusion()
        self._mapper = RetrievalDocumentMapper()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        异步 Hybrid 检索。

        这里向量检索是异步，BM25 是同步。
        """

        vector_hits = await self._vector_retriever.retrieve(
            query=query,
            top_k=self.vector_top_k,
            doc_type=self.doc_type,
            business_domain=self.business_domain,
        )

        bm25_hits = self._bm25_retriever.retrieve(
            query=query,
            top_k=self.bm25_top_k,
            doc_type=self.doc_type,
            business_domain=self.business_domain,
        )

        hybrid_hits = self._fusion.fuse(
            vector_hits=vector_hits,
            bm25_hits=bm25_hits,
            top_k=self.top_k,
        )

        return self._mapper.to_documents(hybrid_hits)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        同步兜底方法。

        工程里建议优先使用 ainvoke。
        """

        return asyncio.run(
            self._aget_relevant_documents(
                query=query,
                run_manager=run_manager,
            )
        )