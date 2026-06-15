# M10 多跳查询分解与结构化证据实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 M9 自适应检索基础上增加复杂问题分解、子问题独立检索、证据配额和结构化 Context，提高多跳与多条件问题的事实覆盖和答案完整性。

**Architecture:** Query Decomposer 只负责生成原子子问题，SubQuery Fusion 负责候选配额与融合，LangGraph 负责编排，Context Builder 负责按子问题渲染证据。简单问题保持原有 M9 链路，模型或分解失败时回退原问题。

**Tech Stack:** Python 3.12、Pydantic、LangGraph、DeepSeek、Elasticsearch、Milvus、Qwen3-Rerank、pytest、MySQL 评测表。

---

### Task 1: 分解领域模型与 Query Decomposer

**Files:**
- Modify: `src/rag_platform/rag/adaptive/models.py`
- Create: `src/rag_platform/rag/adaptive/query_decomposer.py`
- Create: `src/rag_platform/rag/adaptive/query_decomposition_prompt.py`
- Test: `tests/rag/adaptive/test_query_decomposer.py`

- [ ] 先写失败测试：简单问题不调用模型。
- [ ] 先写失败测试：复杂问题返回 2-3 个去重、自包含子问题。
- [ ] 先写失败测试：非法响应、模型异常和单子问题结果回退原问题。
- [ ] 运行测试并确认因实现缺失而失败。
- [ ] 实现触发规则、DeepSeek 结构化调用、Pydantic 校验和确定性回退。
- [ ] 运行测试确认通过。

### Task 2: 子问题候选融合与 Rerank 配额

**Files:**
- Create: `src/rag_platform/rag/adaptive/sub_query_fusion.py`
- Test: `tests/rag/adaptive/test_sub_query_fusion.py`

- [ ] 先写失败测试：每个子问题至少保留一个候选。
- [ ] 先写失败测试：同一 Chunk 跨子问题命中时合并关联和分数。
- [ ] 先写失败测试：Rerank 后恢复子问题最低配额。
- [ ] 运行测试确认失败。
- [ ] 实现子问题 RRF 融合和配额恢复纯函数。
- [ ] 运行测试确认通过。

### Task 3: 接入 LangGraph

**Files:**
- Modify: `src/rag_platform/domain/rag_state.py`
- Modify: `src/rag_platform/rag/graph/rag_retrieval_graph.py`
- Modify: `src/rag_platform/core/config.py`
- Test: `tests/rag/graph/test_rag_retrieval_graph.py`

- [ ] 先写失败测试：简单问题不分解且现有路径不变。
- [ ] 先写失败测试：复杂问题按子问题检索并标记候选元数据。
- [ ] 先写失败测试：统一 Rerank 后每个子问题仍有候选。
- [ ] 先写失败测试：分解模型失败回退原问题。
- [ ] 运行图测试确认失败。
- [ ] 增加 `decompose_query` 节点和依赖注入。
- [ ] 修改检索与融合节点，保存子问题关联和覆盖状态。
- [ ] 保留 M9 最多两轮自适应重试。
- [ ] 运行图测试确认通过。

### Task 4: 结构化 Context 与答案 Prompt

**Files:**
- Modify: `src/rag_platform/rag/context/context_builder.py`
- Modify: `src/rag_platform/application/context_build_service.py`
- Modify: `src/rag_platform/rag/answer/answer_prompt.py`
- Modify: `src/rag_platform/rag/answer/answer_prompt_builder.py`
- Modify: `src/rag_platform/rag/answer/deepseek_answer_generator.py`
- Modify: `src/rag_platform/application/chat_service.py`
- Test: `tests/rag/context/test_context_builder.py`
- Test: `tests/rag/answer/test_answer_prompt_builder.py`
- Test: `tests/application/test_chat_service.py`

- [ ] 先写失败测试：分解请求 Context 按子问题分组。
- [ ] 先写失败测试：共享 Chunk 不生成重复 Citation。
- [ ] 先写失败测试：答案 Prompt 明确要求逐项回答子问题。
- [ ] 先写失败测试：简单问题 Prompt 与 Context 保持兼容。
- [ ] 运行测试确认失败。
- [ ] 扩展 Context Builder 和 Prompt Builder 可选参数。
- [ ] 从 Workflow 将分解计划透传到答案生成器。
- [ ] 运行测试确认通过。

### Task 5: Workflow 与评测可观测性

**Files:**
- Modify: `src/rag_platform/schemas/rag_workflow.py`
- Modify: `src/rag_platform/application/rag_workflow_service.py`
- Modify: `src/rag_platform/evaluation/rag_adapter.py`
- Modify: `scripts/run_rag_evaluation.py`
- Test: `tests/application/test_rag_workflow_service.py`
- Test: `tests/evaluation/test_rag_adapter.py`
- Test: `tests/evaluation/test_run_rag_evaluation_script.py`

- [ ] 先写失败测试：Workflow Response 输出分解和子问题覆盖。
- [ ] 先写失败测试：Retrieval Hit 元数据保留子问题 ID 和文本。
- [ ] 先写失败测试：运行快照包含全部分解配置和 Prompt 版本。
- [ ] 运行测试确认失败。
- [ ] 实现 Schema、Service、Adapter 和配置快照扩展。
- [ ] 增加 CLI `--query-decomposition enabled|disabled`。
- [ ] 运行测试确认通过。

### Task 6: 专项配对评测

**Files:**
- Create: `src/rag_platform/evaluation/query_decomposition_comparison.py`
- Create: `scripts/compare_query_decomposition.py`
- Test: `tests/evaluation/test_query_decomposition_comparison.py`
- Test: `tests/evaluation/test_compare_query_decomposition_script.py`
- Create: `evaluation/reports/m10_query_decomposition_comparison_v2.json`
- Create: `evaluation/reports/m10_query_decomposition_comparison_v2.md`

- [ ] 先写失败测试：统计分解触发率、子问题覆盖率和逐题改善/退化。
- [ ] 先写失败测试：控制组和实验组固定初始 Query 计划。
- [ ] 运行测试确认失败。
- [ ] 实现 MULTI_HOP、MULTI_CONDITION 检索层配对 A/B。
- [ ] 输出 Fact Coverage、Recall、Judge 前置检索指标和延迟。
- [ ] 运行 45 条 DEVELOPMENT 专项评测。

### Task 7: 端到端回归与报告

**Files:**
- Create: `evaluation/reports/M10_E2E_V2_COMPARISON_<date>.md`

- [ ] 使用关闭分解的 M9 配置运行控制组。
- [ ] 使用开启分解的 M10 配置运行实验组。
- [ ] 运行 45 条复杂题端到端 A/B。
- [ ] 若专项无明显退化，再运行 180 条 DEVELOPMENT。
- [ ] 单独运行有效 CLARIFY 校准集。
- [ ] 输出题型、Fact Coverage、Judge、行为、引用和延迟对比。
- [ ] 明确改善 Case、退化 Case、未解决问题和 M10.2/M11 输入。

### Task 8: 最终验证

- [ ] 运行全部新增测试。
- [ ] 运行 `.venv/bin/python -B -m pytest -q`。
- [ ] 运行 `.venv/bin/python -B -m compileall -q src scripts tests`。
- [ ] 运行 `git diff --check`。
- [ ] 检查所有评测 Run 都完整写入逐题结果和 Judge。
