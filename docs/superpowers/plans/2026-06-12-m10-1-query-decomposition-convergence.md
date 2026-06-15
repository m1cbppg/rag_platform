# M10.1 查询分解收敛优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收紧查询分解触发范围，并让复杂查询的动态 Rerank TopN 真正生效，在保留复杂题收益的同时降低全量回归波动和延迟。

**Architecture:** QueryDecomposer 执行确定性门控和模型收益校验，Graph 只在采用分解时计算调用级 Rerank TopN，RerankService 和 Qwen3Reranker 负责向下透传。普通查询、M9 Anchor、自适应重试和异常回退保持不变。

**Tech Stack:** Python 3.12、Pydantic、LangGraph、DeepSeek、Qwen3-Rerank、pytest、MySQL 评测体系。

---

### Task 1: 收紧确定性候选门控

**Files:**
- Modify: `src/rag_platform/rag/adaptive/query_decomposer.py`
- Modify: `tests/rag/adaptive/test_query_decomposer.py`

- [ ] 增加失败测试：单独“如果”不调用模型。
- [ ] 增加失败测试：版本比较和二选一决策不调用模型。
- [ ] 增加失败测试：需要澄清时不调用模型。
- [ ] 运行测试，确认因当前宽松门控而失败。
- [ ] 实现强组合信号、比较意图、二选一和澄清门控。
- [ ] 运行测试确认通过。

### Task 2: 增加模型分解收益校验

**Files:**
- Modify: `src/rag_platform/rag/adaptive/models.py`
- Modify: `src/rag_platform/rag/adaptive/query_decomposer.py`
- Modify: `src/rag_platform/rag/adaptive/query_decomposition_prompt.py`
- Modify: `src/rag_platform/core/config.py`
- Modify: `tests/rag/adaptive/test_query_decomposer.py`

- [ ] 增加失败测试：低收益分解结果被正常跳过。
- [ ] 增加失败测试：`DEPENDENT` 默认回退 M9 链路。
- [ ] 更新现有有效分解测试，提供高收益 `PARALLEL` 输出。
- [ ] 运行测试确认失败原因正确。
- [ ] 增加 `decomposition_type`、`benefit_score` 解析和配置阈值。
- [ ] 更新 Prompt，加入正反例和排除规则。
- [ ] 运行测试确认通过。

### Task 3: 透传动态 Rerank TopN

**Files:**
- Modify: `src/rag_platform/application/rerank_service.py`
- Modify: `src/rag_platform/rag/rerankers/qwen3_reranker.py`
- Modify: `src/rag_platform/rag/graph/rag_retrieval_graph.py`
- Modify: `tests/application/test_core_service_injection.py`
- Modify: `tests/rag/rerankers/test_qwen3_reranker.py`
- Modify: `tests/rag/graph/test_rag_retrieval_graph.py`

- [ ] 增加失败测试：复杂查询传入 `rerank_top_n + 子问题数`。
- [ ] 增加失败测试：简单查询继续使用基础 TopN。
- [ ] 增加失败测试：RerankService 成功和 fail-open 都使用调用级 TopN。
- [ ] 增加失败测试：Qwen3Reranker 将调用级 TopN 传入百炼客户端。
- [ ] 运行测试确认失败。
- [ ] 增加向后兼容的可选 `top_n` 参数并逐层透传。
- [ ] 配额恢复使用相同的有效 TopN。
- [ ] 运行测试确认通过。

### Task 4: 评测快照与报告

**Files:**
- Modify: `scripts/run_rag_evaluation.py`
- Modify: `tests/evaluation/test_run_rag_evaluation_script.py`

- [ ] 增加失败测试：运行快照包含 M10.1 三项配置和新 Prompt 版本。
- [ ] 运行测试确认失败。
- [ ] 扩展评测配置快照。
- [ ] 运行测试确认通过。

### Task 5: 修正组合证据过度拒答

**Files:**
- Modify: `src/rag_platform/rag/answer/action_decision_prompt.py`
- Modify: `tests/application/test_answer_action_decision_service.py`
- Modify: `tests/evaluation/test_run_rag_evaluation_script.py`

- [ ] 增加失败测试：Prompt 允许多个证据块联合覆盖限定条件。
- [ ] 增加失败测试：明确特殊对象例外优先于一般规则。
- [ ] 运行测试确认失败。
- [ ] 更新可回答性 Prompt 和版本。
- [ ] 重跑 False Refusal 多跳题。

### Task 6: 专项评测

**Files:**
- Create: `evaluation/reports/m10_1_query_decomposition_comparison_v2.json`
- Create: `evaluation/reports/m10_1_query_decomposition_comparison_v2.md`
- Create: `evaluation/reports/M10_1_E2E_COMPLEX_20260612.json`
- Create: `evaluation/reports/M10_1_E2E_COMPLEX_20260612.md`

- [ ] 运行 45 条 MULTI_CONDITION 和 MULTI_HOP 检索层配对评测。
- [ ] 运行 45 条复杂题端到端评测。
- [ ] 对比 M10，确认触发、Fact Coverage、Judge 和延迟。
- [ ] 若专项退化，基于逐题归因修正后重跑。

### Task 7: 全量回归

**Files:**
- Create: `evaluation/reports/M10_1_E2E_V2_20260612.json`
- Create: `evaluation/reports/M10_1_E2E_V2_20260612.md`
- Create: `evaluation/reports/M10_1_E2E_V2_COMPARISON_20260612.md`

- [ ] 运行 180 条 DEVELOPMENT。
- [ ] 单独检查 DIRECT、EXACT、MULTI_CONDITION、MULTI_HOP。
- [ ] 检查 Action、REFUSE、CLARIFY 和失败归因。
- [ ] 输出 M9、M10、M10.1 三方对比。

### Task 8: 最终验证

- [ ] 运行 `.venv/bin/python -B -m pytest -q`。
- [ ] 运行 `.venv/bin/python -B -m compileall -q src scripts tests`。
- [ ] 运行 `git diff --check`。
- [ ] 确认所有正式评测 Run 均为完整成功状态。
