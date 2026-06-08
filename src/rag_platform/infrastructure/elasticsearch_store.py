from elasticsearch.helpers import bulk

from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.elasticsearch_client import create_elasticsearch_client


class ElasticsearchChunkStore:
    """
    ES chunk 索引与搜索封装。

    职责：
    1. bulk 写入 chunk；
    2. 按 chunk_id 删除；
    3. BM25 查询。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = create_elasticsearch_client()
        self.index_name = self.settings.es_chunk_index

    def bulk_index_chunks(self, chunks: list[dict]) -> None:
        """
        批量写入 chunk 到 ES。

        使用 ES Bulk API。
        Bulk API 可以在一个请求中执行多个写入动作，减少请求开销。
        """

        if not chunks:
            return

        actions = []

        for chunk in chunks:
            action = {
                "_op_type": "index",
                "_index": self.index_name,

                # 使用 chunk_id 作为 ES _id。
                # 这样重复写入同一个 chunk 时，会覆盖旧文档，而不是新增重复文档。
                "_id": str(chunk["chunk_id"]),

                "_source": chunk,
            }

            actions.append(action)

        bulk(
            client=self.client,
            actions=actions,
            refresh=True,
        )

    def search_bm25(
        self,
        query: str,
        top_k: int,
        doc_type: str | None = None,
        business_domain: str | None = None,
    ) -> list[dict]:
        """
        BM25 检索。

        query：
            用户问题。

        top_k：
            返回数量。

        doc_type / business_domain：
            可选 metadata filter。
        """

        filters = [
            {"term": {"status": "ACTIVE"}}
        ]

        if doc_type:
            filters.append({"term": {"chunk_type": doc_type}})

        if business_domain:
            filters.append({"term": {"business_domain": business_domain}})

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^3",
                                    "title_path^2",
                                    "content",
                                    "keywords^2",
                                    "tags^2"
                                ],
                                "type": "best_fields",
                            }
                        },
                        {
                            "match_phrase": {
                                "content": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        response = self.client.search(
            index=self.index_name,
            body=body,
        )

        hits = response.get("hits", {}).get("hits", [])

        results: list[dict] = []

        for hit in hits:
            source = hit["_source"]

            results.append({
                "chunk_id": int(source["chunk_id"]),
                "score": float(hit["_score"]),
                "source": "bm25",
                "metadata": source,
            })

        return results