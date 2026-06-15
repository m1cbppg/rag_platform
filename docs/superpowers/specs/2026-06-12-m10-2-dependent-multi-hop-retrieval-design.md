# M10.2 顺序依赖多跳检索设计

## 目标

在 M10.1 并行查询分解基础上，增加固定两跳的顺序依赖检索：

```text
原问题
-> 第一跳检索
-> 从第一跳证据抽取中间事实
-> 使用中间事实生成第二跳查询
-> 第二跳检索
-> 两跳证据融合、精排和 Context 构建
```

本阶段只实现两跳，不实现开放式 Agent 循环。

## 适用问题

顺序依赖问题满足以下特征：

1. 第二个检索目标无法仅根据原问题完整确定；
2. 必须先从第一跳证据得到状态、类别、等级、错误含义或当前规则；
3. 第二跳查询必须显式使用第一跳得到的中间事实。

例如：

```text
某错误码表示哪类退款失败？这种失败应走什么处理流程？
```

第一跳查询错误码含义，得到失败类型；第二跳再根据失败类型检索处理流程。

## 数据结构

查询分解继续使用 `QueryDecompositionResult`，但 `DecomposedSubQuery` 增加：

- `depends_on_sub_query_id`：依赖的前置子问题；
- `is_template`：当前问题是否仍包含待填充的中间事实模板。

`DEPENDENT` 分解固定产生两个子问题：

- `SQ1`：可直接执行的第一跳问题；
- `SQ2`：包含 `{{intermediate_fact}}` 的第二跳查询模板。

工作流新增 `dependent_hop`：

- 当前跳数和最大跳数；
- 中间事实；
- 证据原文；
- 支持该事实的 Chunk ID；
- 第二跳实际查询；
- 抽取置信度和是否使用回退。

## 中间事实抽取

新增 `IntermediateFactExtractor`，输入：

- 原始问题；
- 第一跳问题；
- 第二跳查询模板；
- 第一跳 Rerank 后的候选证据。

输出：

- `intermediate_fact`；
- `evidence_quote`；
- `supporting_chunk_id`；
- `confidence`；
- `reason`。

抽取结果必须满足：

1. `supporting_chunk_id` 必须属于第一跳候选；
2. `evidence_quote` 必须是该 Chunk 正文的原文子串；
3. 置信度达到配置阈值；
4. 中间事实非空。

任何校验失败都不能把模型输出写入第二跳查询。

## LangGraph 编排

在 `judge_retrieval_quality` 后增加 `prepare_dependent_hop` 路由：

1. `DEPENDENT` 且第一跳存在候选时，优先进入第二跳准备；
2. 抽取成功时，将中间事实填入 `SQ2` 模板；
3. 检索轮次增加到 2，查询变体标记为 `DEPENDENT_HOP`；
4. 第二跳结果与第一跳结果继续使用现有多轮 RRF 融合；
5. 最终统一 Rerank，并按两个子问题构建结构化 Context。

顺序多跳占用现有最大两轮预算，不再额外执行 M9 第三轮重试。

## 回退策略

- 分解失败：继续原 M9 链路；
- 第一跳无候选：不执行第二跳，交给现有动作决策；
- 中间事实抽取失败：第二跳使用原问题执行一次保守回退检索；
- 第二跳失败：保留第一跳证据继续构建 Context；
- Rerank 或 Context 失败：沿用现有 fail-open 行为。

## 配置

新增：

```text
dependent_multi_hop_enabled=true
dependent_multi_hop_max_hops=2
dependent_fact_model=deepseek-chat
dependent_fact_min_confidence=0.75
dependent_fact_max_candidates=5
```

同时将 `query_decomposition_allow_dependent` 默认值改为 `true`。

## 可观测性

不新增数据库表。

- `decomposition` 保存完整两跳计划；
- `dependent_hop` 保存中间事实及证据来源；
- `retrieval_attempts` 分别保存第一跳和第二跳查询；
- `rag_eval_retrieval_hit.metadata_json` 保存 `sub_query_id` 和跳数；
- `rag_eval_run.config_json` 保存开关、阈值和 Prompt 版本。

## 验收

1. `DEPENDENT` 分解能够执行第一跳和第二跳；
2. 第二跳查询包含第一跳抽取出的中间事实；
3. 中间事实必须有 Chunk 和原文证据支持；
4. 两跳证据都能进入最终融合、Rerank 和 Context；
5. 简单问题和 `PARALLEL` 分解行为不变；
6. 中间事实抽取失败时请求不会失败；
7. 全量自动化测试、编译和差异检查通过。
