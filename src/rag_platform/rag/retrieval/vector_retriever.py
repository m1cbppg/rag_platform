from pymilvus import Collection, connections

from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.search import RetrievalHit
from src.rag_platform.infrastructure.dashscope_embedding import DashScopeEmbeddingClient
from src.rag_platform.infrastructure.repositories.vector_repository import VectorRepository


class VectorRetriever:
    """
    Milvus 向量检索器。

    流程：
    1. 用 text-embedding-v4 把 query 转成 query 向量；
    2. 调用 Milvus search；
    3. 拿 chunk_id 回 MySQL 查完整 chunk metadata；
    4. 转成统一 RetrievalHit。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_client = DashScopeEmbeddingClient()
        self.repository = VectorRepository()

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )

        self.collection = Collection(self.settings.milvus_collection)
        self.collection.load()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
        business_domain: str | None = None,
    ) -> list[RetrievalHit]:
        actual_top_k = top_k or self.settings.rag_top_k

        query_vector = await self.embedding_client.embed_query(query)

        expr_parts = ["status == \"ACTIVE\""]

        if doc_type:
            expr_parts.append(f'doc_type == "{doc_type}"')

        if business_domain:
            expr_parts.append(f'business_domain == "{business_domain}"')

        expr = " and ".join(expr_parts)

        search_params = {
            "metric_type": self.settings.milvus_metric_type,
            "params": {
                "ef": 64
            },
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field=self.settings.milvus_vector_field,
            param=search_params,
            limit=actual_top_k,
            expr=expr,
            output_fields=[
                "chunk_id",
                "doc_id",
                "chunk_code",
                "doc_type",
                "business_domain",
                "version",
                "status",
            ],
        )

        hits = results[0]

        chunk_ids = [
            int(hit.entity.get("chunk_id"))
            for hit in hits
        ]

        chunk_map = self.repository.get_chunks_by_ids(chunk_ids)

        final_hits: list[RetrievalHit] = []

        for hit in hits:
            chunk_id = int(hit.entity.get("chunk_id"))
            chunk = chunk_map.get(chunk_id, {})

            final_hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    score=float(hit.score),
                    source="vector",
                    metadata=chunk,
                )
            )

        return final_hits