# M8.1 回答动作决策设计

## 目标

在现有 RAG 检索、精排和 Context 构建链路后增加统一动作决策，使系统能够明确输出：

- `ANSWER`：证据足够，进入答案生成；
- `REFUSE`：知识库缺少可支持结论的证据，不生成业务答案；
- `CLARIFY`：用户问题缺少必要条件，返回明确追问。

本模块只解决动作选择，不优化召回、精排、引用和答案完整性。

## V0 暴露的问题

正式 V0 共 180 条题：

- 156 条应回答题全部输出 `ANSWER`；
- 18 条应拒答题全部输出 `ANSWER`；
- 6 条应澄清题全部输出 `ANSWER`。

现有 Query 分析已经包含：

```text
need_clarification
clarification_question
```

但这两个字段只保存在 LangGraph State，没有进入工作流响应，也没有影响 Chat 行为。

现有 Chat 仅在 Context 为空时拒答。无关文档同样会形成非空 Context，因此知识缺口题仍会进入答案生成。

## 设计原则

1. 不读取评测用例的 `expected_action`、题型或生成元数据；
2. 不把 24 条动作题的问题文本硬编码为规则；
3. 查询前信号负责识别缺少用户条件；
4. 检索后证据负责识别知识缺口；
5. 动作决策失败时优先保持现有可用性，不因决策模型异常导致 Chat 整体失败；
6. Qwen 继续只负责独立评审，生产动作决策使用 DeepSeek；
7. V0 数据和报告保持不变，优化效果通过新实验版本对照。

## 最终架构

```text
Query 分析
  ↓
混合检索
  ↓
召回质量判断
  ↓
精排
  ↓
Context 构建
  ↓
AnswerActionDecisionService
  ├─ QueryAnalysis 高置信度澄清信号
  ├─ 领域意图/必填槽位策略引擎
  ├─ 精确时间、金额等证据约束守卫
  └─ DeepSeek 两阶段回退
       ├─ 缺失条件判定
       └─ 证据可回答性判定
  ↓
  ├─ ANSWER  → DeepSeekAnswerGenerator
  ├─ REFUSE  → 返回知识不足说明
  └─ CLARIFY → 返回明确追问
```

动作决策器位于 `ChatService` 和检索工作流之间。检索工作流负责暴露完整 Query 分析结果；ChatService 调用决策器并负责三种行为的持久化与响应。

## 数据结构

新增 `AnswerAction`：

```text
ANSWER
REFUSE
CLARIFY
```

新增 `AnswerActionDecision`：

```text
action
confidence
reason
clarification_question
missing_information
decision_source
```

`decision_source` 取值：

- `QUERY_ANALYSIS`：高置信度缺条件；
- `POLICY_ENGINE`：命中领域意图但缺少必填槽位；
- `CONSTRAINT_GUARD`：问题包含 Context 未覆盖的精确限定条件；
- `EMPTY_CONTEXT`：Context 为空；
- `LLM_EVIDENCE`：DeepSeek 根据证据判断；
- `FALLBACK`：模型失败或结果低置信度。

## 决策流程

### 第一级：Query 分析信号

1. Query 分析明确要求澄清；
2. 存在非空 `clarification_question`；
3. Query 分析置信度不低于 `0.75`；

同时满足时直接返回 `CLARIFY`，无需继续调用动作决策模型。

### 第二级：领域槽位策略

配置文件：

```text
config/clarification_policies.json
```

每条策略包含：

- 业务意图触发词组；
- 用户场景触发词；
- 知识型问题排除词；
- 必填槽位及槽位同义表达；
- 标准澄清问题。

仅在“意图匹配、用户正在处理具体场景、必填槽位缺失”时返回
`CLARIFY`。规则说明、版本对比、流程总结等知识型问题继续进入回答链路。

当前覆盖：

- 修改收货地址；
- 取消订单；
- 退款后优惠券返还；
- 仅退款；
- 包裹破损；
- 售后单下一步操作。

### 第三级：空 Context 与精确约束守卫

Context 为空时直接返回 `REFUSE`。

Context 非空时抽取问题中的精确时间、金额、工作日、百分比等约束。
如果问题包含“十分钟”等特殊条件，而 Context 只有一般规则且未出现该条件，
直接返回 `REFUSE`，避免把通用规则错误套用到特殊场景。

### 第四级：DeepSeek 两阶段回退

确定性规则均未命中时，分两次调用 DeepSeek：

1. 缺失条件识别：只判断是否需要用户补充条件；
2. 证据可回答性：只判断 Context 是否支持问题主要结论。

拆分目的：避免模型因“Context 有多个分支可列举”而忽略用户缺少分支条件。

判定要求：

- Context 能直接支持用户问题的主要结论：`ANSWER`；
- Context 只包含相似主题，但缺少问题要求的具体规则：`REFUSE`；
- 用户补充一个或多个条件后，Context 中已有规则可以回答：`CLARIFY`；
- 不能因为 Context 中出现相似关键词就判定可回答；
- 不能用常识补充知识库没有提供的规则。
- 问题中的时间、金额、状态、版本和商品类型等限定条件必须有直接证据。

### 第五级：阈值与降级

- `REFUSE` 只有置信度不低于 `0.80` 才生效；
- `CLARIFY` 只有置信度不低于 `0.75` 且追问非空才生效；
- `ANSWER` 不设置额外阈值；
- 模型超时、格式错误或低置信度时：
  - Context 非空：降级为 `ANSWER`；
  - Context 为空：降级为 `REFUSE`。

该策略优先避免新增大量误拒答。V1_ACTION 评测后再依据混淆矩阵调整阈值。

## Chat 响应

`AnswerStatus` 增加：

```text
CLARIFIED
```

行为映射：

| 动作 | Chat status | 是否调用答案模型 | 是否保留 Citation |
| --- | --- | --- | --- |
| ANSWER | SUCCESS | 是 | 是 |
| REFUSE | REFUSED | 否 | 否 |
| CLARIFY | CLARIFIED | 否 | 否 |

`ChatResponseV2` 增加 `action_decision`，用于诊断和评测复现。

流式接口对 `REFUSE` 和 `CLARIFY` 都直接发送文本，然后发送 `done` 事件。

## 持久化

本阶段不新增数据库表。

`rag_answer_log.status` 为 `VARCHAR(30)`，可以直接保存 `CLARIFIED`。动作决策详情先通过 Chat 响应和评测运行记录保存，不修改现有生产表结构。

## 评测集修正

原 `rag_eval_ecommerce:v1` 中 6 条 CLARIFY 题存在评测契约缺陷：

- 没有来源文档；
- 没有必要事实；
- `required_identifier` 为空；
- 没有明确缺失条件；
- 没有标准追问；
- 部分题实际可被当前知识库完整回答。

因此保留原冻结集和 V0 基线不变，新增独立动作校准集：

```text
rag_eval_action:v1
dataset_id = 28
case_count = 6
sha256 = dfb6a27936563db735e935f8a667fa3f1c3cc8d33818461a9d09b4b12adc8c92
```

每条 CLARIFY 校准题必须包含：

- `missing_condition_key`；
- `missing_condition_label`；
- 标准澄清问题；
- 可接受关键词；
- 至少两个条件分支；
- 每个分支对应的 ACTIVE Chunk 和预期结论。

注册脚本：

```bash
.venv/bin/python -B scripts/register_action_calibration.py
```

Qwen Judge 会读取 `generation_metadata.clarification_contract`，
判断实际追问是否覆盖标准缺失条件。

## 评测适配

`ChatServiceEvaluationAdapter` 增加映射：

```text
SUCCESS    -> ANSWER
REFUSED    -> REFUSE
CLARIFIED  -> CLARIFY
其他状态    -> ERROR
```

正式对照实验：

- 实验版本：`V1_ACTION`
- 实验名称：`action-decision-after-retrieval`
- 数据集：`rag_eval_ecommerce:v1`
- Split：`DEVELOPMENT`

评测 CLI 支持：

```text
--expected-action
--case-code
--limit
```

动作 Prompt、Judge Prompt、策略完整配置和约束守卫版本均写入
`rag_eval_run.config_json`。

## 已完成结果

工程结果：

1. Chat 非流式和流式接口已支持三种动作；
2. 评测 Adapter 已映射 `CLARIFIED -> CLARIFY`；
3. 独立动作校准集已冻结；
4. 原冻结评测集和 V0 基线未修改；
5. 所有策略和 Prompt 版本进入运行快照。

专项结果：

| 运行 | 结果 |
| --- | --- |
| `V1_ACTION_CALIBRATION6_FINAL_20260610` | CLARIFY 6/6，Judge 6/6 |
| `V1_ACTION_REFUSE18_GUARD1_FINAL_20260610` | REFUSE 17/18，Recall 94.44% |
| `V1_ACTION_ANSWER20_GUARD1_20260610` | ANSWER 17/20，3 条均归因到检索或冲突证据不足 |
| `V1_ACTION_CASE007_GUARD1_20260610` | 特殊时间条件拒答 1/1，Judge 通过 |

REFUSE 剩余 1 条失败题的知识库实际包含可回答证据，属于冻结数据集错误金标，
不应通过强制拒答来追求表面指标。
