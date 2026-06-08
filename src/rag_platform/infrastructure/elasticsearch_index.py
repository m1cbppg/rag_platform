from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.elasticsearch_client import create_elasticsearch_client


class ElasticsearchIndexManager:
    """
    Elasticsearch Index 管理器。

    当前只负责：
    1. 判断 index 是否存在；
    2. 创建 chunk BM25 index；
    3. 删除 index，方便本地重建。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = create_elasticsearch_client()
        self.index_name = self.settings.es_chunk_index

    def index_exists(self) -> bool:
        """
        判断 ES index 是否存在。
        """

        return self.client.indices.exists(index=self.index_name)

    def create_index_if_not_exists(self) -> None:
        """
        如果 index 不存在，则创建。

        注意：
        analyzer 使用 IK：
            index analyzer: ik_max_word
            search analyzer: ik_smart

        你已经说明本地 ES 是带 IK 的版本。
        如果没有 IK，这里会创建失败。
        """

        if self.index_exists():
            return

        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "long"},
                    "doc_id": {"type": "long"},
                    "chunk_code": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "business_domain": {"type": "keyword"},
                    "version": {"type": "keyword"},
                    "status": {"type": "keyword"},

                    "title": {
                        "type": "text",
                        "analyzer": self.settings.es_analyzer,
                        "search_analyzer": self.settings.es_search_analyzer,
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "title_path": {
                        "type": "text",
                        "analyzer": self.settings.es_analyzer,
                        "search_analyzer": self.settings.es_search_analyzer,
                    },
                    "content": {
                        "type": "text",
                        "analyzer": self.settings.es_analyzer,
                        "search_analyzer": self.settings.es_search_analyzer,
                    },
                    "keywords": {
                        "type": "text",
                        "analyzer": self.settings.es_analyzer,
                        "search_analyzer": self.settings.es_search_analyzer,
                    },
                    "tags": {
                        "type": "text",
                        "analyzer": self.settings.es_analyzer,
                        "search_analyzer": self.settings.es_search_analyzer,
                    },
                    "source_section": {
                        "type": "keyword"
                    }
                }
            }
        }

        self.client.indices.create(
            index=self.index_name,
            body=body,
        )

    def delete_index(self) -> None:
        """
        删除 ES index。

        本地开发可用。
        生产环境要非常谨慎，必须加权限。
        """

        if self.index_exists():
            self.client.indices.delete(index=self.index_name)