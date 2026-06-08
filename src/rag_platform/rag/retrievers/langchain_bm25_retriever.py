from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from src.rag_platform.core.config import get_settings
from src.rag_platform.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag_platform.rag.retrievers.document_mapper import RetrievalDocumentMapper


class LangChainBM25Retriever(BaseRetriever):
    """
    LangChain BM25 Retriever。

    作用：
    把模块 6 的 BM25Retriever 包装成 LangChain BaseRetriever。

    注意：
    BaseRetriever 是 Pydantic 模型，所以字段要用类型注解声明。
    不要像普通 Python 类那样随便在 __init__ 里塞属性。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_k: int = Field(default=20, description="BM25 返回数量")
    doc_type: str | None = Field(default=None, description="文档类型过滤")
    business_domain: str | None = Field(default=None, description="业务域过滤")

    _bm25_retriever: BM25Retriever = PrivateAttr()
    _mapper: RetrievalDocumentMapper = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic v2 模型初始化后会调用这个方法。

        这里初始化真正的底层 BM25Retriever。
        这样 API 不会在 import 时创建 ES 连接，而是在实例化 Retriever 时创建。
        """

        self._bm25_retriever = BM25Retriever()
        self._mapper = RetrievalDocumentMapper()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        LangChain BaseRetriever 的同步检索方法。

        query：
            用户问题。

        返回：
            list[Document]
        """

        hits = self._bm25_retriever.retrieve(
            query=query,
            top_k=self.top_k,
            doc_type=self.doc_type,
            business_domain=self.business_domain,
        )

        return self._mapper.to_documents(hits)