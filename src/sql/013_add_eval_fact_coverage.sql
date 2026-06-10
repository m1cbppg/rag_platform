-- 适用数据库：MySQL 8.x
-- 用途：为逐题评测结果增加必要事实覆盖率，支持多条件和多跳问题诊断。
-- 执行前提：已经执行 010_create_rag_evaluation_tables.sql。

ALTER TABLE rag_eval_case_result
    ADD COLUMN fact_coverage DECIMAL(8,6) DEFAULT NULL
        COMMENT '必要事实覆盖率：召回结果覆盖的fact_key数量除以全部必要fact_key数量'
        AFTER ndcg_at_10;
