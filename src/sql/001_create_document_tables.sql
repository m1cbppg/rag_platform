CREATE TABLE IF NOT EXISTS rag_document (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',

    doc_code VARCHAR(64) NOT NULL UNIQUE COMMENT '文档唯一编号',
    title VARCHAR(255) NOT NULL COMMENT '文档标题',

    doc_type VARCHAR(50) NOT NULL COMMENT '文档类型：FAQ/SOP/RULE/MANUAL',
    file_name VARCHAR(255) NOT NULL COMMENT '原始文件名',
    file_path VARCHAR(500) NOT NULL COMMENT '服务器本地保存路径',
    file_ext VARCHAR(20) NOT NULL COMMENT '文件扩展名：docx/pdf',

    business_domain VARCHAR(100) DEFAULT NULL COMMENT '业务域，例如订单、支付、用户、客服流程',
    version VARCHAR(50) DEFAULT NULL COMMENT '文档版本',

    status VARCHAR(30) NOT NULL DEFAULT 'UPLOADED' COMMENT '状态：UPLOADED/PARSING/CLEANED/NEED_REVIEW/FAILED',

    created_by VARCHAR(64) DEFAULT NULL COMMENT '上传人',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    INDEX idx_doc_type (doc_type),
    INDEX idx_business_domain (business_domain),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG文档主表';


CREATE TABLE IF NOT EXISTS rag_document_parse (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',

    doc_id BIGINT NOT NULL COMMENT '文档ID',
    parser_type VARCHAR(100) NOT NULL COMMENT '解析器类型',

    raw_content LONGTEXT COMMENT '原始解析文本',
    clean_content LONGTEXT COMMENT '清洗后的文本',
    structure_json JSON COMMENT '结构化解析结果',

    parse_status VARCHAR(30) NOT NULL DEFAULT 'SUCCESS' COMMENT 'SUCCESS/FAILED/NEED_REVIEW',
    error_message TEXT DEFAULT NULL COMMENT '错误信息',

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    INDEX idx_doc_id (doc_id),

    CONSTRAINT fk_document_parse_doc
        FOREIGN KEY (doc_id) REFERENCES rag_document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG文档解析结果表';


CREATE TABLE IF NOT EXISTS rag_document_quality (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',

    doc_id BIGINT NOT NULL COMMENT '文档ID',
    check_item VARCHAR(100) NOT NULL COMMENT '校验项',
    check_result VARCHAR(20) NOT NULL COMMENT 'PASS/WARN/FAIL',
    message TEXT DEFAULT NULL COMMENT '校验说明',

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

    INDEX idx_doc_id (doc_id),
    INDEX idx_check_result (check_result),

    CONSTRAINT fk_document_quality_doc
        FOREIGN KEY (doc_id) REFERENCES rag_document(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG文档质量校验表';