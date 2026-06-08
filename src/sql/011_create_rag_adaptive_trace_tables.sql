-- 适用数据库：MySQL 8.x
-- 用途：记录自适应检索运行过程，以及Claim级答案证据校验结果。

CREATE TABLE IF NOT EXISTS rag_retrieval_trace (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '自适应检索追踪主键ID',
    trace_id VARCHAR(64) NOT NULL COMMENT '一次RAG请求的全链路追踪ID',
    session_id VARCHAR(128) DEFAULT NULL COMMENT '会话ID',
    original_question TEXT NOT NULL COMMENT '用户原始问题',
    final_rewritten_question TEXT DEFAULT NULL COMMENT '最终采用的改写问题',
    selected_route VARCHAR(30) DEFAULT NULL
        COMMENT '最终检索路由：BM25、VECTOR向量、HYBRID混合、MULTI_HOP多跳',
    expected_action VARCHAR(30) DEFAULT NULL
        COMMENT '系统预判行为：ANSWER回答、REFUSE拒答、CLARIFY追问澄清',
    actual_action VARCHAR(30) DEFAULT NULL
        COMMENT '系统最终实际行为：ANSWER回答、REFUSE拒答、CLARIFY追问澄清、ERROR错误',
    max_rounds INT NOT NULL DEFAULT 2 COMMENT '允许执行的最大检索轮数',
    completed_rounds INT NOT NULL DEFAULT 0 COMMENT '实际完成的检索轮数',
    final_quality_level VARCHAR(20) DEFAULT NULL
        COMMENT '最终召回质量等级：GOOD良好、WEAK较弱、POOR不足',
    final_quality_score DECIMAL(8,6) DEFAULT NULL COMMENT '最终召回质量综合分，范围0到1',
    status VARCHAR(30) NOT NULL DEFAULT 'STARTED'
        COMMENT '链路状态：STARTED已开始、SUCCESS成功、FAILED失败',
    error_message TEXT DEFAULT NULL COMMENT '链路级错误信息',
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '链路开始时间',
    finished_at DATETIME DEFAULT NULL COMMENT '链路结束时间',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_retrieval_trace_id (trace_id),
    KEY idx_retrieval_trace_status (status),
    KEY idx_retrieval_trace_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自适应检索请求级运行追踪表';


CREATE TABLE IF NOT EXISTS rag_retrieval_attempt (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '单轮检索尝试主键ID',
    retrieval_trace_id BIGINT NOT NULL COMMENT '所属自适应检索追踪ID',
    round_no INT NOT NULL COMMENT '检索轮次，从1开始',
    strategy VARCHAR(50) NOT NULL
        COMMENT '本轮策略：INITIAL初始、QUERY_REWRITE查询改写、RELAX_FILTER放宽过滤、MULTI_QUERY多查询、SUB_QUERY子问题、HYDE假设文档',
    query_variant VARCHAR(30) NOT NULL
        COMMENT '查询变体类型，例如ORIGINAL、REWRITTEN、SUB_QUERY',
    query_text TEXT NOT NULL COMMENT '本轮实际用于检索的查询文本',
    retrieval_mode VARCHAR(30) NOT NULL COMMENT '本轮检索模式：bm25、vector或hybrid',
    doc_type_filter VARCHAR(50) DEFAULT NULL COMMENT '本轮文档类型过滤条件',
    business_domain_filter VARCHAR(100) DEFAULT NULL COMMENT '本轮业务域过滤条件',
    top_k INT NOT NULL COMMENT '本轮每个召回通道的目标召回数量',
    candidate_count INT NOT NULL DEFAULT 0 COMMENT '本轮去重后的候选Chunk数量',
    quality_level VARCHAR(20) DEFAULT NULL COMMENT '本轮召回质量等级：GOOD、WEAK、POOR',
    quality_score DECIMAL(8,6) DEFAULT NULL COMMENT '本轮召回质量综合分，范围0到1',
    quality_features_json JSON DEFAULT NULL COMMENT '候选数、通道重合率、精排分数等质量特征JSON',
    trigger_reason TEXT DEFAULT NULL COMMENT '触发本轮检索或重试的原因',
    latency_ms INT DEFAULT NULL COMMENT '本轮检索耗时，单位毫秒',
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        COMMENT '本轮状态：PENDING待执行、RUNNING执行中、SUCCESS成功、FAILED失败',
    error_message TEXT DEFAULT NULL COMMENT '本轮错误信息',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_retrieval_attempt_round (
        retrieval_trace_id,
        round_no,
        query_variant
    ),
    KEY idx_retrieval_attempt_quality (quality_level, quality_score),
    CONSTRAINT fk_retrieval_attempt_trace
        FOREIGN KEY (retrieval_trace_id) REFERENCES rag_retrieval_trace(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自适应检索每轮尝试及质量决策表';


CREATE TABLE IF NOT EXISTS rag_retrieval_hit_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '运行时召回明细主键ID',
    retrieval_attempt_id BIGINT NOT NULL COMMENT '所属单轮检索尝试ID',
    channel VARCHAR(30) NOT NULL
        COMMENT '召回通道：BM25、VECTOR向量、HYBRID融合、RERANK精排、FINAL最终结果',
    chunk_id BIGINT NOT NULL COMMENT '召回的rag_chunk.id',
    channel_rank INT DEFAULT NULL COMMENT '该召回通道中的原始排名',
    final_rank INT DEFAULT NULL COMMENT '多轮融合或精排后的最终排名',
    raw_score DECIMAL(18,10) DEFAULT NULL COMMENT '召回通道原始分数',
    fused_score DECIMAL(18,10) DEFAULT NULL COMMENT 'RRF等融合策略得到的分数',
    rerank_score DECIMAL(18,10) DEFAULT NULL COMMENT '精排模型给出的相关性分数',
    selected_for_context TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否进入最终Context：1是、0否',
    discard_reason VARCHAR(255) DEFAULT NULL COMMENT '未进入Context时的淘汰原因',
    metadata_json JSON DEFAULT NULL COMMENT '过滤条件、查询来源、事实标识等诊断元数据JSON',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    UNIQUE KEY uk_retrieval_hit_log (
        retrieval_attempt_id,
        channel,
        chunk_id
    ),
    KEY idx_retrieval_hit_final_rank (retrieval_attempt_id, final_rank),
    KEY idx_retrieval_hit_chunk (chunk_id),
    CONSTRAINT fk_retrieval_hit_attempt
        FOREIGN KEY (retrieval_attempt_id) REFERENCES rag_retrieval_attempt(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='运行时召回排名及Context选择明细表';


CREATE TABLE IF NOT EXISTS rag_grounding_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '答案证据校验主键ID',
    trace_id VARCHAR(64) NOT NULL COMMENT '对应RAG请求的追踪ID',
    answer_log_id BIGINT DEFAULT NULL
        COMMENT '对应rag_answer_log.id，仅逻辑关联',
    validator_provider VARCHAR(50) NOT NULL COMMENT '证据校验模型提供方',
    validator_model VARCHAR(100) NOT NULL COMMENT '证据校验模型名称',
    prompt_version VARCHAR(30) NOT NULL COMMENT '证据校验Prompt版本',
    claim_count INT NOT NULL DEFAULT 0 COMMENT '答案拆分出的可验证Claim总数',
    supported_claim_count INT NOT NULL DEFAULT 0 COMMENT '证据完全支持的Claim数量',
    unsupported_claim_count INT NOT NULL DEFAULT 0 COMMENT '证据不支持的Claim数量',
    conflicting_claim_count INT NOT NULL DEFAULT 0 COMMENT '证据之间存在冲突的Claim数量',
    support_ratio DECIMAL(8,6) DEFAULT NULL COMMENT '被支持Claim占全部Claim的比例',
    answer_confidence DECIMAL(8,6) DEFAULT NULL COMMENT '综合答案可信度，范围0到1',
    decision VARCHAR(30) NOT NULL
        COMMENT '校验决策：PASS通过、CORRECT修正、REFUSE拒答、FAILED校验失败',
    correction_applied TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否执行过答案修正：1是、0否',
    latency_ms INT DEFAULT NULL COMMENT '证据校验及修正耗时，单位毫秒',
    status VARCHAR(30) NOT NULL DEFAULT 'SUCCESS'
        COMMENT '处理状态：SUCCESS成功、FAILED失败',
    error_message TEXT DEFAULT NULL COMMENT '证据校验错误信息',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    KEY idx_grounding_trace (trace_id),
    KEY idx_grounding_answer_log (answer_log_id),
    KEY idx_grounding_decision (decision)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Claim级答案证据校验汇总表';


CREATE TABLE IF NOT EXISTS rag_claim_evidence_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT 'Claim证据明细主键ID',
    grounding_log_id BIGINT NOT NULL COMMENT '所属答案证据校验记录ID',
    claim_no INT NOT NULL COMMENT 'Claim在答案中的顺序编号',
    claim_text TEXT NOT NULL COMMENT '从答案中抽取的原子化可验证声明',
    claim_type VARCHAR(30) DEFAULT NULL
        COMMENT 'Claim类型：FACT事实、RULE规则、PROCEDURE流程、NUMBER数值、CONCLUSION结论',
    citation_ids_json JSON DEFAULT NULL COMMENT '该Claim使用的引用编号列表JSON，例如C1、C2',
    chunk_ids_json JSON DEFAULT NULL COMMENT '该Claim引用的Chunk ID列表JSON',
    support_status VARCHAR(30) NOT NULL
        COMMENT '证据支持状态：SUPPORTED支持、PARTIAL部分支持、UNSUPPORTED不支持、CONFLICT冲突',
    support_score DECIMAL(8,6) DEFAULT NULL COMMENT '证据对Claim的支持度，范围0到1',
    evidence_reason TEXT DEFAULT NULL COMMENT '支持、不支持或冲突的判断理由',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    UNIQUE KEY uk_claim_evidence (grounding_log_id, claim_no),
    KEY idx_claim_support_status (support_status),
    CONSTRAINT fk_claim_evidence_grounding
        FOREIGN KEY (grounding_log_id) REFERENCES rag_grounding_log(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='答案原子Claim及其引用证据明细表';
