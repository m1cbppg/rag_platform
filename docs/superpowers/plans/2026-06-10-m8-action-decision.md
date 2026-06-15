# M8.1 回答动作决策实施计划

> **执行要求：** 使用 TDD，先写失败测试，再实现最小生产代码。当前工作区直接实施，不创建独立 worktree。

**目标：** 在检索后增加统一 `ANSWER/REFUSE/CLARIFY` 动作决策，并建立 `V1_ACTION` 对照结果。

**架构：** Query 分析、配置化领域槽位策略、精确约束守卫和 DeepSeek 两阶段回退共同完成动作决策。ChatService 根据结果选择生成答案、拒答或追问，Qwen 继续用于独立评审。

**技术栈：** Python 3.12、Pydantic、DeepSeek Chat JSON、LangGraph、pytest、MySQL。

---

### 任务 1：动作决策数据结构与 Service

**文件：**

- 新建：`src/rag_platform/domain/answer_action.py`
- 新建：`src/rag_platform/schemas/answer_action.py`
- 新建：`src/rag_platform/rag/answer/action_decision_prompt.py`
- 新建：`src/rag_platform/application/answer_action_decision_service.py`
- 新建：`tests/application/test_answer_action_decision_service.py`
- 修改：`src/rag_platform/core/config.py`

- [x] 测试高置信度 `need_clarification` 直接返回 `CLARIFY`。
- [x] 测试空 Context 直接返回 `REFUSE`。
- [x] 实现 DeepSeek 缺失条件和可回答性两阶段判断。
- [x] 测试低置信度拒答降级为 `ANSWER`。
- [x] 测试模型异常时有 Context 降级 `ANSWER`、无 Context 降级 `REFUSE`。
- [x] 增加动作决策开关、模型和阈值配置。
- [x] 实现严格 Pydantic 校验和决策来源记录。

运行：

```bash
.venv/bin/python -B -m pytest \
  tests/application/test_answer_action_decision_service.py -q
```

### 任务 2：工作流透传 Query 分析

**文件：**

- 修改：`src/rag_platform/schemas/rag_workflow.py`
- 修改：`src/rag_platform/application/rag_workflow_service.py`
- 修改：`tests/application/test_rag_workflow_service.py`

- [x] 测试响应包含 `query_analysis`。
- [x] 测试响应包含 `need_clarification` 和 `clarification_question`。
- [x] 保持现有检索、精排和 Context 字段兼容。

运行：

```bash
.venv/bin/python -B -m pytest \
  tests/application/test_rag_workflow_service.py -q
```

### 任务 3：ChatService 三路行为

**文件：**

- 修改：`src/rag_platform/domain/answer.py`
- 修改：`src/rag_platform/schemas/chat_v2.py`
- 修改：`src/rag_platform/application/chat_service.py`
- 修改：`tests/application/test_chat_service.py`

- [x] 测试 `ANSWER` 继续调用答案生成器。
- [x] 测试 `REFUSE` 不调用答案生成器、不保存 Citation。
- [x] 测试 `CLARIFY` 返回追问且状态为 `CLARIFIED`。
- [x] 测试动作决策详情写入 Chat 响应。
- [x] 测试流式接口的 `REFUSE/CLARIFY` 文本和 `done` 状态。
- [x] 抽取统一的直接响应保存逻辑，避免拒答和澄清重复实现。

运行：

```bash
.venv/bin/python -B -m pytest \
  tests/application/test_chat_service.py -q
```

### 任务 4：评测 Adapter 动作映射

**文件：**

- 修改：`src/rag_platform/evaluation/rag_adapter.py`
- 修改：`tests/evaluation/test_rag_adapter.py`

- [x] 测试 `CLARIFIED` 映射为 `ActualAction.CLARIFY`。
- [x] 确认 `REFUSED` 仍映射为 `REFUSE`。
- [x] 确认非成功状态映射为 `ERROR`。

运行：

```bash
.venv/bin/python -B -m pytest \
  tests/evaluation/test_rag_adapter.py -q
```

### 任务 5：运行配置快照

**文件：**

- 修改：`scripts/run_rag_evaluation.py`
- 修改：`tests/evaluation/test_run_rag_evaluation_script.py`

- [x] 将动作决策开关、模型、阈值和 Prompt 版本写入 `config_json`。
- [x] 写入领域策略完整快照、Judge Prompt 和约束守卫版本。
- [x] 增加 `--expected-action`、`--case-code` 精确筛选。

运行：

```bash
.venv/bin/python -B -m pytest \
  tests/evaluation/test_run_rag_evaluation_script.py -q
```

### 任务 6：动作校准集与确定性守卫

**文件：**

- 新建：`config/clarification_policies.json`
- 新建：`src/rag_platform/application/clarification_policy_service.py`
- 新建：`src/rag_platform/application/evidence_constraint_service.py`
- 新建：`src/rag_platform/evaluation/action_calibration.py`
- 新建：`scripts/register_action_calibration.py`
- 新建：`evaluation/datasets/rag_action_calibration_v1.jsonl`

- [x] 定义 source-backed CLARIFY 契约。
- [x] 校验标准缺失条件、标准追问和 ACTIVE Chunk 分支。
- [x] 注册并冻结 `rag_eval_action:v1`。
- [x] 实现配置化领域意图/必填槽位策略。
- [x] 实现精确时间、金额等证据约束守卫。
- [x] Qwen Judge 按澄清契约评审追问质量。

### 任务 7：专项验证与 V1_ACTION

- [x] CLARIFY 校准集：6/6，Judge 6/6。
- [x] REFUSE：17/18，Recall 94.44%。
- [x] 特殊时间条件拒答：1/1，Judge 通过。
- [x] ANSWER 抽样：17/20；失败题均归因到检索或冲突证据不足。
- [x] 静态扫描 156 条 ANSWER：策略提前拦截 0 条。
- [ ] 完整 180 条 `V1_ACTION` 留到检索和冲突证据优化完成后执行。
- [ ] 生成完整 Markdown 和 JSON 对照报告。

正式运行标识：

```text
experiment_version = V1_ACTION
experiment_name = action-decision-after-retrieval
run_code = V1_ACTION_DEVELOPMENT_20260610
```

最终验证：

```bash
.venv/bin/python -B -m pytest -q
RUN_MYSQL_INTEGRATION=1 .venv/bin/python -B -m pytest \
  tests/evaluation/test_dataset_repository.py -q
.venv/bin/python -m compileall -q src scripts tests
git diff --check
```
