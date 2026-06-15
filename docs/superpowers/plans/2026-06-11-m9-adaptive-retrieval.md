# M9 自适应检索实施计划

> 按任务逐步实施，每一步先写测试，再实现并验证。

**Goal:** 将固定单轮检索升级为可校准、最多两轮、全程可追踪的自适应检索。

**Architecture:** 质量特征提取器只计算信号，质量策略只负责决策，Query Rewriter 负责生成第二轮查询，多轮融合器负责合并各轮候选。LangGraph 编排重试循环，工作流响应和评测适配器输出每轮明细。

**Tech Stack:** Python、Pydantic、LangGraph、DeepSeek、Elasticsearch、Milvus、Qwen3-Rerank、pytest、MySQL 评测表。

---

### Task 1: 自适应检索领域模型和质量特征

**Files:**
- Create: `src/rag_platform/rag/adaptive/__init__.py`
- Create: `src/rag_platform/rag/adaptive/models.py`
- Create: `src/rag_platform/rag/adaptive/quality_features.py`
- Test: `tests/rag/adaptive/test_quality_features.py`

- [ ] 编写失败测试，覆盖候选数、文档数、通道重合率、精确词覆盖率和版本数。
- [ ] 运行测试，确认因模块不存在而失败。
- [ ] 实现纯函数特征提取器。
- [ ] 运行测试，确认通过。

### Task 2: 质量决策策略

**Files:**
- Create: `src/rag_platform/rag/adaptive/quality_policy.py`
- Modify: `src/rag_platform/core/config.py`
- Test: `tests/rag/adaptive/test_quality_policy.py`

- [ ] 编写失败测试，覆盖无候选、精确词缺失、版本不足、目标类型缺失和低相关度。
- [ ] 运行测试，确认失败。
- [ ] 增加自适应检索开关、最大轮数、阈值和策略版本配置。
- [ ] 实现 GOOD/WEAK/POOR 和重试策略决策。
- [ ] 运行测试，确认通过。

### Task 3: Query Rewrite

**Files:**
- Create: `src/rag_platform/rag/adaptive/query_rewriter.py`
- Create: `src/rag_platform/rag/adaptive/query_rewrite_prompt.py`
- Test: `tests/rag/adaptive/test_query_rewriter.py`

- [ ] 编写失败测试，覆盖模型成功、非法响应和模型异常兜底。
- [ ] 运行测试，确认失败。
- [ ] 使用注入式 DeepSeek Client 实现结构化查询改写。
- [ ] 限制候选摘要长度和最多 3 条查询。
- [ ] 实现不依赖模型的确定性兜底。
- [ ] 运行测试，确认通过。

### Task 4: 多轮结果融合

**Files:**
- Create: `src/rag_platform/rag/adaptive/multi_round_fusion.py`
- Modify: `src/rag_platform/rag/retrievers/document_mapper.py`
- Test: `tests/rag/adaptive/test_multi_round_fusion.py`

- [ ] 编写失败测试，覆盖轮次权重、Chunk 去重和来源元数据保留。
- [ ] 运行测试，确认失败。
- [ ] 实现加权 RRF。
- [ ] 修复 Document Mapper 丢弃 Hybrid 排名元数据的问题。
- [ ] 运行测试，确认通过。

### Task 5: 接入 LangGraph

**Files:**
- Modify: `src/rag_platform/domain/rag_state.py`
- Modify: `src/rag_platform/rag/graph/rag_retrieval_graph.py`
- Modify: `tests/rag/graph/test_rag_retrieval_graph.py`

- [ ] 编写失败测试，证明 GOOD 不重试、WEAK 重试一次、第二轮后停止。
- [ ] 编写失败测试，证明 FORCE_BM25 和 RELAX_FILTER 修改正确的检索参数。
- [ ] 运行测试，确认失败。
- [ ] 调整流程为检索、融合、精排、质量判断、条件重试。
- [ ] 保存每轮查询、过滤条件、候选和质量信息。
- [ ] 运行图测试，确认通过。

### Task 6: 工作流响应和评测追踪

**Files:**
- Modify: `src/rag_platform/schemas/rag_workflow.py`
- Modify: `src/rag_platform/application/rag_workflow_service.py`
- Modify: `src/rag_platform/evaluation/rag_adapter.py`
- Modify: `tests/application/test_rag_workflow_service.py`
- Modify: `tests/evaluation/test_rag_adapter.py`

- [ ] 编写失败测试，要求响应输出检索轮次和每轮尝试。
- [ ] 编写失败测试，要求评测 Hit 使用真实轮次和 Query 变体。
- [ ] 运行测试，确认失败。
- [ ] 扩展响应 Schema 和映射逻辑。
- [ ] 将每轮召回、最终 Rerank 和 Context 结果写入评测 Hit。
- [ ] 运行测试，确认通过。

### Task 7: 实验配置和校准脚本

**Files:**
- Modify: `scripts/run_rag_evaluation.py`
- Create: `scripts/calibrate_retrieval_quality.py`
- Modify: `tests/evaluation/test_run_rag_evaluation_script.py`
- Create: `tests/evaluation/test_calibrate_retrieval_quality.py`

- [ ] 编写失败测试，要求配置快照包含全部自适应参数。
- [ ] 编写失败测试，覆盖阈值候选评分目标。
- [ ] 运行测试，确认失败。
- [ ] 扩展实验配置快照。
- [ ] 实现只允许 DEVELOPMENT/VALIDATION 数据的阈值校准脚本。
- [ ] 运行测试，确认通过。

### Task 8: 验证和定向评测

- [ ] 运行新增自适应检索测试。
- [ ] 运行完整测试套件。
- [ ] 运行 MySQL 集成测试。
- [ ] 定向运行地址修改、精确编号、冲突和多跳失败 Case。
- [ ] 对比每个 Case 的首轮和最终 Recall、事实覆盖率、动作和延迟。
- [ ] 校准开发集阈值，并在验证集选择配置。

### Task 9: V1 开发集评测和报告

- [ ] 使用冻结开发集运行完整 V1。
- [ ] 生成 V0/V1 对比报告。
- [ ] 记录 Recall@K、MRR、nDCG、事实覆盖率、动作准确率和 Judge 指标。
- [ ] 记录二次检索触发率、策略分布和 p50/p95 延迟增幅。
- [ ] 明确本阶段修复 Case、退化 Case和下一阶段多跳分解输入。

### Task 10: 修正 EXACT identifier Gold 证据

**Files:**
- Create: `src/rag_platform/evaluation/exact_evidence_correction.py`
- Create: `scripts/create_corrected_eval_dataset.py`
- Create: `tests/evaluation/test_exact_evidence_correction.py`
- Create: `tests/evaluation/test_create_corrected_eval_dataset.py`
- Create: `evaluation/datasets/rag_eval_v2.frozen.jsonl`
- Create: `evaluation/reports/rag_eval_v2_correction.json`

- [ ] 编写失败测试，覆盖缺失 identifier、新增事实、已正确 Case 不变和无法定位时报错。
- [ ] 运行测试，确认失败。
- [ ] 实现只在原 source_doc_codes 中检索的确定性校正。
- [ ] 实现 v2 数据集注册、源文档复制、Case/Evidence 落库和冻结。
- [ ] 审计全部 EXACT Case，确保 required_identifier 在至少一个必要证据 Chunk 中。
- [ ] 在 v2 上重新运行 V0 与 V1，确保比较使用同一数据集 SHA。

### Task 11: 成对检索专项评测

**Files:**
- Create: `src/rag_platform/evaluation/adaptive_retrieval_comparison.py`
- Create: `scripts/compare_adaptive_retrieval.py`
- Create: `tests/evaluation/test_adaptive_retrieval_comparison.py`
- Create: `tests/evaluation/test_compare_adaptive_retrieval_script.py`
- Create: `evaluation/reports/m9_adaptive_retrieval_comparison_v2.json`
- Create: `evaluation/reports/m9_adaptive_retrieval_comparison_v2.md`

- [ ] 编写失败测试，覆盖均值差、改善/退化计数、触发率和策略分布。
- [ ] 编写失败测试，证明控制组和实验组强制使用相同的规则 Query 计划。
- [ ] 实现逐题成对运行，不调用 Answer Generator 和 Judge。
- [ ] 保存控制组/实验组的轮次、策略、最终 Chunk、指标和延迟。
- [ ] 生成 JSON 与中文 Markdown 报告。
- [ ] 在 v2 DEVELOPMENT 的全部 ANSWER Case 上运行。
- [ ] 记录显著改善、无变化和退化 Case，作为下一阶段优化输入。
