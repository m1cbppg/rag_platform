from elasticsearch import Elasticsearch

from src.rag_platform.core.config import get_settings


def create_elasticsearch_client() -> Elasticsearch:
    """
    创建 Elasticsearch 客户端。

    你当前本地 ES 可以直接 curl http://localhost:9200，
    说明本地大概率没有开启认证，所以 username/password 可以为空。

    如果以后 ES 开启认证，只需要在 .env 里配置：
        ELASTICSEARCH_USERNAME=elastic
        ELASTICSEARCH_PASSWORD=xxx
    """

    settings = get_settings()

    if settings.elasticsearch_username and settings.elasticsearch_password:
        return Elasticsearch(
            hosts=[settings.elasticsearch_url],
            basic_auth=(
                settings.elasticsearch_username,
                settings.elasticsearch_password,
            ),
            verify_certs=settings.elasticsearch_verify_certs,
        )

    return Elasticsearch(
        hosts=[settings.elasticsearch_url],
        verify_certs=settings.elasticsearch_verify_certs,
    )