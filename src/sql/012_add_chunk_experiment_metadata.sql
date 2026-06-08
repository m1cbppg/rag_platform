-- 适用数据库：MySQL 8.x
-- 用途：为Chunking消融实验增加策略及实验隔离字段。
-- 执行前请先检查当前表结构；如果某个字段已经存在，应删除对应ADD COLUMN语句后再执行。

ALTER TABLE rag_chunk
    ADD COLUMN chunk_strategy VARCHAR(50) NOT NULL DEFAULT 'STRUCTURED'
        COMMENT '切分策略：FIXED固定长度、RECURSIVE递归、STRUCTURED结构化、PARENT_CHILD父子、SEMANTIC语义切分'
        AFTER chunk_type,
    ADD COLUMN chunk_strategy_version VARCHAR(30) NOT NULL DEFAULT 'v1'
        COMMENT '切分策略版本，用于区分同一策略的不同实现'
        AFTER chunk_strategy,
    ADD COLUMN chunk_size INT DEFAULT NULL
        COMMENT '目标Chunk大小；结构化切分不适用时可为空'
        AFTER chunk_strategy_version,
    ADD COLUMN chunk_overlap INT DEFAULT NULL
        COMMENT '相邻Chunk重叠大小；无重叠时为0或空'
        AFTER chunk_size,
    ADD COLUMN parser_version VARCHAR(30) DEFAULT NULL
        COMMENT '生成该Chunk时使用的文档解析器版本'
        AFTER chunk_overlap,
    ADD COLUMN content_sha256 CHAR(64) DEFAULT NULL
        COMMENT 'Chunk规范化内容SHA256摘要，用于幂等及实验映射'
        AFTER parser_version,
    ADD INDEX idx_chunk_strategy (
        doc_id,
        chunk_strategy,
        chunk_strategy_version,
        status
    ),
    ADD INDEX idx_chunk_content_sha256 (content_sha256);


ALTER TABLE rag_embedding_task
    ADD COLUMN experiment_namespace VARCHAR(64) NOT NULL DEFAULT 'production'
        COMMENT 'Embedding实验命名空间，用于隔离不同切分或索引实验，production表示正常业务数据'
        AFTER milvus_collection,
    ADD INDEX idx_embedding_experiment (
        experiment_namespace,
        status
    );


ALTER TABLE rag_keyword_index_task
    ADD COLUMN experiment_namespace VARCHAR(64) NOT NULL DEFAULT 'production'
        COMMENT 'Elasticsearch索引实验命名空间，用于隔离不同切分或检索实验'
        AFTER index_name,
    ADD INDEX idx_keyword_index_experiment (
        experiment_namespace,
        status
    );
