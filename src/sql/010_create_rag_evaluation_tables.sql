-- 适用数据库：MySQL 8.x
-- 用途：创建 RAG 评测子系统所需的数据表。
-- 执行前提：核心 RAG 业务表已经创建完成。

CREATE TABLE IF NOT EXISTS rag_eval_dataset (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '评测数据集主键ID',
    dataset_code VARCHAR(64) NOT NULL COMMENT '数据集业务编码，例如 rag_eval_ecommerce',
    name VARCHAR(255) NOT NULL COMMENT '数据集名称',
    version VARCHAR(32) NOT NULL COMMENT '数据集版本，例如 v1',
    domain VARCHAR(100) NOT NULL COMMENT '业务领域，例如 ecommerce_after_sales',
    description TEXT DEFAULT NULL COMMENT '数据集说明',
    generator_provider VARCHAR(50) DEFAULT NULL COMMENT '语料生成模型提供方，例如 deepseek',
    generator_model VARCHAR(100) DEFAULT NULL COMMENT '语料生成模型名称',
    reviewer_provider VARCHAR(50) DEFAULT NULL COMMENT '独立审核模型提供方，例如 dashscope',
    reviewer_model VARCHAR(100) DEFAULT NULL COMMENT '独立审核模型名称',
    status VARCHAR(30) NOT NULL DEFAULT 'DRAFT'
        COMMENT '数据集状态：DRAFT草稿、GENERATED已生成、REVIEWED已审核、FROZEN已冻结、ARCHIVED已归档',
    document_count INT NOT NULL DEFAULT 0 COMMENT '数据集包含的源文档数量',
    case_count INT NOT NULL DEFAULT 0 COMMENT '数据集包含的评测题数量',
    generation_config_json JSON DEFAULT NULL COMMENT '语料及题目生成配置JSON',
    content_sha256 CHAR(64) DEFAULT NULL COMMENT '冻结数据集规范化内容的SHA256摘要',
    frozen_at DATETIME DEFAULT NULL COMMENT '数据集冻结时间',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_eval_dataset_code_version (dataset_code, version),
    KEY idx_eval_dataset_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG评测数据集版本表';


CREATE TABLE IF NOT EXISTS rag_eval_source_document (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '评测源文档主键ID',
    dataset_id BIGINT NOT NULL COMMENT '所属评测数据集ID',
    source_doc_code VARCHAR(64) NOT NULL COMMENT '源文档业务编码，生成、导入及映射过程保持不变',
    title VARCHAR(255) NOT NULL COMMENT '源文档标题',
    doc_type VARCHAR(30) NOT NULL COMMENT '文档类型：FAQ问答、SOP流程、RULE规则、MANUAL操作手册',
    topic VARCHAR(100) NOT NULL COMMENT '文档主题，例如 refund、coupon、logistics',
    version VARCHAR(50) DEFAULT NULL COMMENT '文档版本',
    effective_from DATE DEFAULT NULL COMMENT '文档或规则生效日期',
    effective_to DATE DEFAULT NULL COMMENT '文档或规则失效日期，为空表示未设定失效日期',
    is_current TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否为当前有效版本：1是、0否',
    relative_file_path VARCHAR(500) NOT NULL COMMENT '生成语料文件相对路径',
    source_content_sha256 CHAR(64) NOT NULL COMMENT '生成源内容SHA256摘要',
    generation_spec_json JSON DEFAULT NULL COMMENT '该文档的生成蓝图及约束JSON',
    review_status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        COMMENT '审核状态：PENDING待审核、PASSED通过、REJECTED拒绝',
    review_score DECIMAL(6,4) DEFAULT NULL COMMENT '独立模型给出的文档审核总分，范围0到1',
    review_reason TEXT DEFAULT NULL COMMENT '审核结论、问题及拒绝原因',
    mapped_doc_id BIGINT DEFAULT NULL
        COMMENT '文档导入后对应的rag_document.id，仅逻辑关联，不设置外键',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_eval_source_doc (dataset_id, source_doc_code),
    KEY idx_eval_source_type (dataset_id, doc_type),
    KEY idx_eval_source_topic (dataset_id, topic),
    KEY idx_eval_source_mapped_doc (mapped_doc_id),
    CONSTRAINT fk_eval_source_dataset
        FOREIGN KEY (dataset_id) REFERENCES rag_eval_dataset(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评测数据集生成源文档表';


CREATE TABLE IF NOT EXISTS rag_eval_case (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '评测题主键ID',
    dataset_id BIGINT NOT NULL COMMENT '所属评测数据集ID',
    case_code VARCHAR(64) NOT NULL COMMENT '评测题业务编码',
    question TEXT NOT NULL COMMENT '用户问题原文',
    normalized_question TEXT DEFAULT NULL COMMENT '用于去重和检索分析的规范化问题',
    reference_answer LONGTEXT DEFAULT NULL COMMENT '参考答案，仅允许评测Judge使用，不得传入被测RAG',
    case_type VARCHAR(50) NOT NULL
        COMMENT '题型：DIRECT直接问答、PARAPHRASE同义模糊、EXACT精确标识、MULTI_CONDITION多条件、MULTI_HOP多跳、CONFLICT冲突版本、NO_ANSWER无答案',
    target_doc_types_json JSON DEFAULT NULL COMMENT '预期检索的文档类型列表JSON',
    expected_action VARCHAR(30) NOT NULL DEFAULT 'ANSWER'
        COMMENT '预期行为：ANSWER回答、REFUSE拒答、CLARIFY追问澄清',
    difficulty VARCHAR(20) NOT NULL DEFAULT 'MEDIUM'
        COMMENT '难度：EASY简单、MEDIUM中等、HARD困难',
    dataset_split VARCHAR(20) NOT NULL DEFAULT 'UNASSIGNED'
        COMMENT '数据划分：DEVELOPMENT开发集、VALIDATION验证集、TEST测试集、UNASSIGNED未分配',
    business_domain VARCHAR(100) DEFAULT NULL COMMENT '题目所属业务域',
    required_fact_count INT NOT NULL DEFAULT 1 COMMENT '正确回答所需的必要事实数量',
    generation_metadata_json JSON DEFAULT NULL COMMENT '题目生成来源、参数及过程元数据JSON',
    review_status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        COMMENT '审核状态：PENDING待审核、PASSED通过、REJECTED拒绝',
    review_score DECIMAL(6,4) DEFAULT NULL COMMENT '独立模型给出的题目审核总分，范围0到1',
    review_reason TEXT DEFAULT NULL COMMENT '题目审核结论、问题及拒绝原因',
    status VARCHAR(30) NOT NULL DEFAULT 'ACTIVE'
        COMMENT '题目状态：ACTIVE有效、DISABLED停用',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_eval_case (dataset_id, case_code),
    KEY idx_eval_case_type (dataset_id, case_type),
    KEY idx_eval_case_action (dataset_id, expected_action),
    KEY idx_eval_case_split (dataset_id, dataset_split),
    KEY idx_eval_case_review (dataset_id, review_status),
    CONSTRAINT fk_eval_case_dataset
        FOREIGN KEY (dataset_id) REFERENCES rag_eval_dataset(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG评测题及参考答案表';


CREATE TABLE IF NOT EXISTS rag_eval_case_relevance (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '标准证据关联主键ID',
    case_id BIGINT NOT NULL COMMENT '所属评测题ID',
    source_document_id BIGINT NOT NULL COMMENT '标准证据所在评测源文档ID',
    mapped_doc_id BIGINT DEFAULT NULL
        COMMENT '导入后对应的rag_document.id，仅逻辑关联',
    mapped_chunk_id BIGINT DEFAULT NULL
        COMMENT '切分后对应的rag_chunk.id，仅逻辑关联',
    relevance_grade TINYINT NOT NULL DEFAULT 1
        COMMENT '相关度等级：0不相关、1相关、2高度相关、3回答所必需的证据',
    evidence_quote TEXT DEFAULT NULL COMMENT '从源文档逐字复制的标准证据片段',
    evidence_quote_sha256 CHAR(64) DEFAULT NULL COMMENT '标准证据规范化文本SHA256摘要',
    fact_key VARCHAR(100) DEFAULT NULL
        COMMENT '事实标识，用于多跳检索的必要事实覆盖率及答案完整性评测',
    mapping_status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        COMMENT '证据映射状态：PENDING待映射、MAPPED已映射、AMBIGUOUS存在歧义、MISSING未找到',
    mapping_reason TEXT DEFAULT NULL COMMENT '证据映射结果说明或失败原因',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_eval_case_evidence (
        case_id,
        source_document_id,
        evidence_quote_sha256
    ),
    KEY idx_eval_relevance_case_chunk (case_id, mapped_chunk_id),
    KEY idx_eval_relevance_mapping (mapping_status),
    CONSTRAINT fk_eval_relevance_case
        FOREIGN KEY (case_id) REFERENCES rag_eval_case(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_eval_relevance_source
        FOREIGN KEY (source_document_id) REFERENCES rag_eval_source_document(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评测题标准文档、Chunk及证据片段表';


CREATE TABLE IF NOT EXISTS rag_eval_run (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '评测运行主键ID',
    run_code VARCHAR(64) NOT NULL COMMENT '评测运行唯一业务编码',
    dataset_id BIGINT NOT NULL COMMENT '本次运行使用的数据集ID',
    experiment_version VARCHAR(30) NOT NULL
        COMMENT '实验版本：V0、V1、V2、V3或自定义实验标签',
    experiment_name VARCHAR(255) NOT NULL COMMENT '实验名称',
    git_commit_sha VARCHAR(64) DEFAULT NULL COMMENT '被测代码Git提交SHA',
    retrieval_mode VARCHAR(30) DEFAULT NULL COMMENT '主要检索模式',
    embedding_model VARCHAR(100) DEFAULT NULL COMMENT 'Embedding模型名称',
    rerank_model VARCHAR(100) DEFAULT NULL COMMENT 'Rerank模型名称',
    answer_model VARCHAR(100) DEFAULT NULL COMMENT '答案生成模型名称',
    judge_model VARCHAR(100) DEFAULT NULL COMMENT '独立评审模型名称',
    config_json JSON NOT NULL COMMENT '本次实验完整配置快照JSON',
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        COMMENT '运行状态：PENDING待执行、RUNNING执行中、SUCCESS成功、PARTIAL部分成功、FAILED失败、CANCELLED取消',
    total_cases INT NOT NULL DEFAULT 0 COMMENT '计划执行的评测题总数',
    completed_cases INT NOT NULL DEFAULT 0 COMMENT '已完成评测题数量',
    failed_cases INT NOT NULL DEFAULT 0 COMMENT '执行失败的评测题数量',
    summary_metrics_json JSON DEFAULT NULL COMMENT '整次实验汇总指标JSON',
    started_at DATETIME DEFAULT NULL COMMENT '运行开始时间',
    finished_at DATETIME DEFAULT NULL COMMENT '运行结束时间',
    error_message TEXT DEFAULT NULL COMMENT '运行级错误信息',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_eval_run_code (run_code),
    KEY idx_eval_run_dataset (dataset_id, experiment_version),
    KEY idx_eval_run_status (status),
    CONSTRAINT fk_eval_run_dataset
        FOREIGN KEY (dataset_id) REFERENCES rag_eval_dataset(id)
        ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='可复现的RAG评测实验运行表';


CREATE TABLE IF NOT EXISTS rag_eval_case_result (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '逐题评测结果主键ID',
    run_id BIGINT NOT NULL COMMENT '所属评测运行ID',
    case_id BIGINT NOT NULL COMMENT '对应评测题ID',
    trace_id VARCHAR(64) DEFAULT NULL COMMENT '被测RAG请求链路追踪ID',
    actual_action VARCHAR(30) DEFAULT NULL
        COMMENT '系统实际行为：ANSWER回答、REFUSE拒答、CLARIFY追问、ERROR错误',
    generated_answer LONGTEXT DEFAULT NULL COMMENT '被测RAG生成的答案',
    retrieved_chunk_ids_json JSON DEFAULT NULL COMMENT '最终有序召回Chunk ID列表JSON',
    cited_chunk_ids_json JSON DEFAULT NULL COMMENT '答案实际引用Chunk ID列表JSON',
    recall_at_1 DECIMAL(8,6) DEFAULT NULL COMMENT 'Recall@1召回率',
    recall_at_3 DECIMAL(8,6) DEFAULT NULL COMMENT 'Recall@3召回率',
    recall_at_5 DECIMAL(8,6) DEFAULT NULL COMMENT 'Recall@5召回率',
    recall_at_10 DECIMAL(8,6) DEFAULT NULL COMMENT 'Recall@10召回率',
    reciprocal_rank DECIMAL(8,6) DEFAULT NULL COMMENT '首个相关结果倒数排名MRR单题值',
    ndcg_at_5 DECIMAL(8,6) DEFAULT NULL COMMENT 'nDCG@5排序质量指标',
    ndcg_at_10 DECIMAL(8,6) DEFAULT NULL COMMENT 'nDCG@10排序质量指标',
    citation_precision DECIMAL(8,6) DEFAULT NULL COMMENT '引用精确率',
    citation_recall DECIMAL(8,6) DEFAULT NULL COMMENT '引用召回率',
    action_correct TINYINT(1) DEFAULT NULL COMMENT '实际回答、拒答或澄清行为是否正确：1正确、0错误',
    retrieval_rounds INT NOT NULL DEFAULT 1 COMMENT '实际执行的检索轮数',
    input_tokens INT DEFAULT NULL COMMENT '本题模型输入Token总数',
    output_tokens INT DEFAULT NULL COMMENT '本题模型输出Token总数',
    estimated_cost DECIMAL(14,8) DEFAULT NULL COMMENT '本题估算调用费用',
    latency_ms INT DEFAULT NULL COMMENT '本题端到端耗时，单位毫秒',
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        COMMENT '逐题状态：PENDING待执行、SUCCESS成功、FAILED失败、SKIPPED跳过',
    error_message TEXT DEFAULT NULL COMMENT '逐题执行错误信息',
    started_at DATETIME DEFAULT NULL COMMENT '本题执行开始时间',
    finished_at DATETIME DEFAULT NULL COMMENT '本题执行结束时间',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_eval_case_result (run_id, case_id),
    KEY idx_eval_case_result_status (run_id, status),
    KEY idx_eval_case_result_trace (trace_id),
    CONSTRAINT fk_eval_case_result_run
        FOREIGN KEY (run_id) REFERENCES rag_eval_run(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_eval_case_result_case
        FOREIGN KEY (case_id) REFERENCES rag_eval_case(id)
        ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='单道评测题的确定性指标及生成结果表';


CREATE TABLE IF NOT EXISTS rag_eval_retrieval_hit (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '评测召回明细主键ID',
    case_result_id BIGINT NOT NULL COMMENT '所属逐题评测结果ID',
    retrieval_round INT NOT NULL DEFAULT 1 COMMENT '检索轮次，从1开始',
    query_variant VARCHAR(30) NOT NULL DEFAULT 'ORIGINAL'
        COMMENT '查询变体：ORIGINAL原问题、REWRITTEN改写、EXPANDED扩展、SUB_QUERY子问题、HYDE假设文档',
    query_text TEXT NOT NULL COMMENT '本轮实际用于检索的查询文本',
    channel VARCHAR(30) NOT NULL
        COMMENT '结果通道：BM25、VECTOR向量、HYBRID融合、RERANK精排、FINAL最终结果',
    chunk_id BIGINT NOT NULL COMMENT '召回的rag_chunk.id',
    rank_no INT NOT NULL COMMENT '该通道中的排名，从1开始',
    raw_score DECIMAL(18,10) DEFAULT NULL COMMENT '检索通道原始分数',
    fused_score DECIMAL(18,10) DEFAULT NULL COMMENT '融合后的分数',
    rerank_score DECIMAL(18,10) DEFAULT NULL COMMENT '精排相关性分数',
    is_gold TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否属于标准相关Chunk：1是、0否',
    metadata_json JSON DEFAULT NULL COMMENT '召回来源、过滤条件及其他诊断元数据JSON',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    UNIQUE KEY uk_eval_retrieval_hit (
        case_result_id,
        retrieval_round,
        query_variant,
        channel,
        chunk_id
    ),
    KEY idx_eval_hit_rank (case_result_id, channel, rank_no),
    KEY idx_eval_hit_chunk (chunk_id),
    CONSTRAINT fk_eval_hit_case_result
        FOREIGN KEY (case_result_id) REFERENCES rag_eval_case_result(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用于检索诊断及消融实验的逐条召回排名表';


CREATE TABLE IF NOT EXISTS rag_eval_judge_result (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '模型评审结果主键ID',
    case_result_id BIGINT NOT NULL COMMENT '所属逐题评测结果ID',
    judge_provider VARCHAR(50) NOT NULL COMMENT '评审模型提供方',
    judge_model VARCHAR(100) NOT NULL COMMENT '评审模型名称',
    judge_prompt_version VARCHAR(30) NOT NULL COMMENT '评审Prompt版本',
    faithfulness_score DECIMAL(6,4) DEFAULT NULL COMMENT '忠实度评分，答案是否仅基于上下文，范围0到1',
    answer_relevance_score DECIMAL(6,4) DEFAULT NULL COMMENT '答案相关性评分，范围0到1',
    completeness_score DECIMAL(6,4) DEFAULT NULL COMMENT '答案必要事实完整性评分，范围0到1',
    citation_entailment_score DECIMAL(6,4) DEFAULT NULL COMMENT '引用对结论的支持度评分，范围0到1',
    conflict_handling_score DECIMAL(6,4) DEFAULT NULL COMMENT '规则或版本冲突处理评分，范围0到1',
    refusal_correct TINYINT(1) DEFAULT NULL COMMENT '拒答行为是否正确：1正确、0错误',
    clarification_correct TINYINT(1) DEFAULT NULL COMMENT '追问澄清行为是否正确：1正确、0错误',
    passed TINYINT(1) NOT NULL DEFAULT 0 COMMENT '模型评审是否总体通过：1通过、0不通过',
    reason_json JSON NOT NULL COMMENT '评审理由、缺失事实和不支持声明JSON',
    raw_response_json JSON DEFAULT NULL COMMENT '评审模型原始响应JSON',
    latency_ms INT DEFAULT NULL COMMENT '评审模型调用耗时，单位毫秒',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    UNIQUE KEY uk_eval_judge_result (
        case_result_id,
        judge_provider,
        judge_model,
        judge_prompt_version
    ),
    KEY idx_eval_judge_passed (passed),
    CONSTRAINT fk_eval_judge_case_result
        FOREIGN KEY (case_result_id) REFERENCES rag_eval_case_result(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='独立LLM-as-a-Judge评审评分表';
