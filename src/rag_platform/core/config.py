from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings 是整个项目的配置类。

    所有环境变量都会在这里统一管理。
    后续业务代码不要直接读取 os.getenv，而是通过 get_settings() 获取。
    """

    app_name: str = "rag-platform"
    app_env: str = "local"
    app_debug: bool = True

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "rag_platform"

    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "rag_chunk_vector"

    # 模块 4 新增：Milvus 向量字段和索引配置
    milvus_vector_field: str = "embedding"
    milvus_metric_type: str = "COSINE"
    milvus_index_type: str = "HNSW"
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"

    # 模块 4 新增：阿里 DashScope Embedding 配置
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    dashscope_chat_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_judge_model: str = "qwen-plus"
    qwen_judge_timeout_seconds: int = 120
    eval_pdf_font_path: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024
    embedding_output_type: str = "dense"
    dashscope_embedding_endpoint: str = "/api/v1/services/embeddings/text-embedding/text-embedding"
    embedding_batch_size: int = 10
    embedding_timeout_seconds: int = 60
    embedding_max_retries: int = 3

    upload_dir: str = "storage/uploads"

    rag_top_k: int = 20
    rag_rerank_top_k: int = 5
    rag_context_max_tokens: int = 6000

    # --------------------
    # Elasticsearch 配置
    # --------------------
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    elasticsearch_verify_certs: bool = False

    es_chunk_index: str = "rag_chunk_bm25"
    es_analyzer: str = "ik_max_word"
    es_search_analyzer: str = "ik_smart"
    es_bm25_top_k: int = 20
    es_index_batch_size: int = 100

    # --------------------
    # Hybrid Search 配置
    # --------------------
    hybrid_vector_weight: float = 0.6
    hybrid_bm25_weight: float = 0.4
    hybrid_final_top_k: int = 20

    # --------------------
    # Query 理解配置
    # --------------------
    query_analysis_use_llm: bool = True
    query_analysis_llm_timeout_seconds: int = 30
    query_analysis_min_confidence: float = 0.6

    default_retrieval_mode: str = "hybrid"
    default_query_top_k: int = 10

    # --------------------
    # Hybrid Fusion 配置
    # --------------------
    hybrid_fusion_method: str = "rrf"
    rrf_rank_constant: int = 60
    rrf_window_size: int = 50

    # --------------------
    # Rerank 配置
    # --------------------
    rerank_enabled: bool = True
    rerank_provider: str = "dashscope"
    rerank_model: str = "qwen3-rerank"
    dashscope_rerank_base_url: str = "https://dashscope.aliyuncs.com"
    dashscope_rerank_endpoint: str = "/compatible-api/v1/reranks"
    rerank_top_n: int = 5
    rerank_candidate_limit: int = 50
    rerank_timeout_seconds: int = 60
    rerank_max_retries: int = 2
    rerank_fail_open: bool = True
    rerank_min_score: float = 0.0
    rerank_instruct: str = (
        "Given a user question, retrieve relevant enterprise knowledge passages "
        "that directly answer the question."
    )

    # --------------------
    # Context 构建配置
    # --------------------
    context_max_tokens: int = 6000
    context_max_chunks: int = 8
    context_expand_parent: bool = True
    context_expand_previous_next: bool = True
    context_expand_same_section: bool = True
    context_max_expanded_chunks_per_hit: int = 2
    context_include_metadata_header: bool = True
    context_citation_prefix: str = "C"

    # --------------------
    # Answer Generation 配置
    # --------------------
    answer_model: str = "deepseek-v4-pro"
    answer_temperature: float = 0.2
    answer_max_tokens: int = 2048
    answer_timeout_seconds: int = 120
    answer_require_citation: bool = True
    answer_fail_when_context_empty: bool = True

    # --------------------
    # SSE 配置
    # --------------------
    chat_stream_heartbeat_seconds: int = 15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置对象。

    lru_cache 会缓存 Settings 对象，避免每次调用都重新读取 .env。
    """
    return Settings()
