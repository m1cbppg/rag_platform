# M6 RAG Evaluation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把冻结评测集、现有 RAG Chat 链路、M5 确定性指标和百炼 Qwen Judge 连接成可续跑的自动化实验执行器。

**Architecture:** 保持现有检索和答案行为不变，在 `ChatService` 内增加不暴露给 API 的可观测执行结果。实验执行器按题调用 Chat、计算确定性指标、调用独立 Judge，并通过 `DatasetRepository` 保存逐题结果和运行汇总。外部模型、数据库和 RAG 都通过协议注入，以支持无基础设施单元测试。

**Tech Stack:** Python 3.12、Pydantic、asyncio、SQLAlchemy、MySQL 8、DeepSeek、DashScope Qwen、pytest。

---

### Task 1: Chat 可观测执行结果

**Files:**
- Create: `src/rag_platform/evaluation/rag_adapter.py`
- Modify: `src/rag_platform/application/chat_service.py`
- Test: `tests/application/test_chat_service.py`
- Test: `tests/evaluation/test_rag_adapter.py`

- [ ] 先写失败测试，要求 Chat 内部执行结果同时包含 API 响应、Workflow Response 和端到端耗时。
- [ ] 把现有 `chat()` 主体下沉到 `execute()`，`chat()` 继续只返回 `ChatResponseV2`，保持对外 API 不变。
- [ ] 实现 Evaluation Adapter：映射实际行为、最终 Context Chunk、有序引用 Chunk、融合和精排诊断 Hit。
- [ ] 验证 REFUSED、SUCCESS、ERROR 三条路径；当前系统不伪造 CLARIFY。

### Task 2: Qwen Answer Judge

**Files:**
- Create: `src/rag_platform/evaluation/judge_service.py`
- Create: `evaluation/prompts/answer_judge.txt`
- Test: `tests/evaluation/test_judge_service.py`

- [ ] 先写 Judge JSON 契约、严格解析和有限重试测试。
- [ ] ANSWER 题按忠实度、相关性、完整性、引用蕴含和冲突处理评分；REFUSE/CLARIFY 使用对应布尔结论。
- [ ] 默认阈值设为 0.8，距阈值不超过 0.05 的分数触发一次证据逆序复评，最终分数取平均。
- [ ] 保存所有原始响应、Prompt 版本和调用耗时。

### Task 3: Repository 续跑能力

**Files:**
- Modify: `src/rag_platform/evaluation/dataset_repository.py`
- Test: `tests/evaluation/test_dataset_repository.py`

- [ ] 增加按 `run_code` 查找运行的方法。
- [ ] 增加幂等准备逐题结果的方法：成功题跳过，失败或未完成题清理子记录后重跑。
- [ ] Judge 保存改为 upsert。
- [ ] 增加运行结果查询，供汇总和恢复使用。

### Task 4: 实验执行器

**Files:**
- Create: `src/rag_platform/evaluation/experiment_runner.py`
- Test: `tests/evaluation/test_experiment_runner.py`

- [ ] 使用 Fake RAG、Fake Judge 和 Fake Repository 写失败测试。
- [ ] 单题流程：准备结果记录、调用 RAG、构建 Gold、计算 M5 指标、保存 Hit、调用 Judge、完成结果。
- [ ] 单题异常转为 `ActualAction.ERROR`，保存失败结果，但不能终止其他题。
- [ ] 使用 `asyncio.Semaphore`，并发只允许 1 到 3。
- [ ] 运行结束根据失败数量写入 SUCCESS、PARTIAL 或 FAILED。

### Task 5: 汇总指标

**Files:**
- Create: `src/rag_platform/evaluation/run_summary.py`
- Test: `tests/evaluation/test_run_summary.py`

- [ ] 对非空指标求平均，明确排除 NO_ANSWER 的 `None` 检索指标。
- [ ] 输出整体指标、按题型指标、行为混淆矩阵、Judge 通过率、延迟 p50/p95。
- [ ] ERROR 独立计数，不能归入拒答。

### Task 6: CLI 和真实冒烟

**Files:**
- Create: `scripts/run_rag_evaluation.py`
- Test: `tests/evaluation/test_run_rag_evaluation_script.py`

- [ ] 支持 dataset、split、experiment version/name、run code、top-k、concurrency 和 limit。
- [ ] 创建运行时保存 Git SHA、冻结数据集 SHA、模型、索引、Collection、Prompt 版本和检索参数。
- [ ] 支持使用相同 `run_code` 续跑。
- [ ] 先执行 Fake 全链路测试，再在 DEVELOPMENT 分片运行 1 条真实冒烟题。
- [ ] 运行全量测试、MySQL 集成测试和 `git diff --check`。
