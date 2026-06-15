# M10 多跳查询分解与结构化证据设计

## 目标

在 M9 自适应检索基础上，解决复杂问题只覆盖部分必要事实，以及检索已命中
多个事实但答案没有逐项组织的问题。

本阶段实现：

1. 复杂问题识别与原子子问题分解；
2. 每个子问题独立检索；
3. 子问题证据配额与跨子问题融合；
4. 按子问题组织 Context；
5. 引导答案模型逐项回答；
6. 分解过程和子问题覆盖情况可观测。

本阶段不实现版本优先级和生效时间冲突消解，也不实现答案生成后的 Claim 级
事实校验。这两项分别属于 M10.2 和 M11。

## 当前问题

M9 端到端评测中：

- MULTI_CONDITION Fact Coverage 为 83.33%，Judge 通过率为 70.37%；
- MULTI_HOP Fact Coverage 为 80.56%，Judge 通过率仅为 27.78%；
- MULTI_HOP Citation Recall 仅为 33.33%；
- M9 已显著减少普通漏召回，剩余问题主要是多事实覆盖和答案组织。

现有 Query Rewrite 只改变检索表达，不知道一个问题包含多少个独立信息需求。
现有 Rerank 也按整个问题统一排序，可能让某个子问题的高分候选挤掉其他子问题
的必要证据。

## 设计原则

1. 简单问题继续走 M9 链路，不承担分解模型调用和额外检索成本。
2. 分解最多产生 3 个原子子问题。
3. 子问题必须自包含，不能使用“上述情况”“这种规则”等指代表达。
4. 每个子问题独立检索，并在候选元数据中保留 `sub_query_id`。
5. 子问题之间先保证最低证据配额，再按综合排名填充剩余候选。
6. 分解失败时回退原问题，不能使原有 RAG 请求失败。
7. M9 自适应二次检索保留；分解和重试总轮数仍不超过两轮。
8. 不读取评测题型或 Gold fact_key，线上触发不能依赖评测元数据。

## 方案

### 1. Query Decomposer

新增 `QueryDecomposer`，输出：

```json
{
  "requires_decomposition": true,
  "reason": "问题包含两个独立信息需求",
  "sub_queries": [
    {
      "sub_query_id": "SQ1",
      "question": "未出库订单修改地址需要满足什么条件？",
      "target_doc_types": ["FAQ", "MANUAL"]
    },
    {
      "sub_query_id": "SQ2",
      "question": "待审核售后单需要上传什么材料？",
      "target_doc_types": ["FAQ", "SOP"]
    }
  ]
}
```

触发采用“确定性候选判断 + DeepSeek 结构化确认”：

- 包含两个以上问号或明显并列问句；
- 包含“如果……同时……”“分别”“以及”“并且”等组合关系；
- 包含多个操作目标、规则对象或条件分支；
- 比较类问题暂时只标记复杂，不在本阶段实现冲突优先级判断。

确定性判断只决定是否值得调用分解模型，不直接生成最终子问题。

### 2. 子问题检索

`analyze_query` 后增加 `decompose_query` 节点。

- 不需要分解：保留现有 `retrieval_queries`。
- 需要分解：每个子问题成为一个检索任务。
- 每个任务携带 `sub_query_id`、子问题文本和目标文档类型。
- Retriever 返回的每个文档写入：
  - `sub_query_ids`
  - `sub_query_texts`
  - `retrieval_query`

同一个 Chunk 可以同时支持多个子问题，元数据必须合并，不能因去重而丢失关联。

### 3. 子问题证据融合

新增 `SubQueryFusion`：

1. 每个子问题按原始检索排名生成 RRF 分数；
2. 每个子问题优先保留至少 1 个候选；
3. 相同 Chunk 跨子问题命中时累加分数；
4. 剩余位置按综合分数填充；
5. 最终候选数量仍受 `rerank_candidate_limit` 控制。

分解候选经过统一 Rerank，但 Rerank 后再次执行子问题最低配额恢复：

- 优先保留每个子问题的最高分候选；
- 再按 Rerank 分数补齐到 `rerank_top_n`；
- 防止统一 Rerank 再次淘汰某个子问题的全部证据。

### 4. 子问题覆盖

新增 `SubQueryCoverage`：

- `total_sub_queries`
- `covered_sub_queries`
- `coverage_rate`
- 每个子问题的候选数和最终 Context Chunk 数

覆盖判定只表示“该子问题至少保留一个候选”，不声称该候选一定是 Gold 证据。
真正的事实正确性仍由冻结评测集和 Qwen Judge 判断。

### 5. 结构化 Context

Context Builder 接收可选的子问题列表。

分解请求渲染为：

```text
## 子问题 SQ1：未出库订单修改地址需要满足什么条件？

[C1] ...
[C2] ...

## 子问题 SQ2：待审核售后单需要上传什么材料？

[C3] ...
```

同时支持“共享证据”：如果一个 Chunk 支持多个子问题，只生成一个 Citation，
但可以在相关子问题标题下引用同一个 Citation。

简单问题继续使用原 Context 格式。

### 6. 答案生成

Workflow Response 增加：

- `decomposition`
- `sub_query_coverage`

Answer Prompt 增加可选的子问题计划：

- 必须逐个回答所有子问题；
- 每个子问题的结论必须携带 Citation；
- 某个子问题证据不足时单独说明，不能忽略后继续给出完整结论。

### 7. M9 自适应检索协作

第一轮按子问题检索后，仍执行 M9 质量判断。

- 整体低相关：继续 `QUERY_REWRITE`；
- 精确编号缺失：继续 `FORCE_BM25`；
- 子问题覆盖不足：新增 `QUERY_DECOMPOSE_RETRY` 原因，但复用
  `QUERY_REWRITE` 策略，只对未覆盖子问题重写；
- 第二轮结果与第一轮结果继续使用 M9 多轮 RRF 融合。

本阶段不增加第三轮。

## 配置

新增：

```text
query_decomposition_enabled=true
query_decomposition_model=deepseek-chat
query_decomposition_max_sub_queries=3
query_decomposition_max_attempts=2
query_decomposition_min_query_length=18
sub_query_min_candidates=1
sub_query_rerank_quota=1
```

评测 CLI 增加 `--query-decomposition enabled|disabled`，并将配置与 Prompt
版本写入 `rag_eval_run.config_json`。

## 可观测性

不新增数据库表。通过现有结构保存：

- State 和 Workflow Response：分解结果与覆盖情况；
- `rag_eval_retrieval_hit.metadata_json`：
  `sub_query_ids`、`sub_query_texts`；
- `rag_eval_run.config_json`：分解开关、模型、阈值和 Prompt 版本；
- 报告：分解触发率、子问题覆盖率、延迟增幅、改善和退化 Case。

## 失败处理

- 模型超时或异常：回退原问题，不分解；
- JSON 非法：最多重试配置次数，然后回退；
- 子问题为空或重复：规范化、去重，少于 2 个则不分解；
- 单个子问题检索失败：保留其他子问题结果；
- 全部子问题失败：回退原问题执行现有 M9 检索；
- 结构化 Context 构建失败：回退普通 Context。

## 验收

专项 A/B 使用 v2 DEVELOPMENT 中 MULTI_HOP 和 MULTI_CONDITION：

- MULTI_HOP Fact Coverage 目标不低于 90%；
- MULTI_CONDITION Fact Coverage 目标不低于 90%；
- 两类 Judge 通过率高于 M9；
- 输出分解触发率、子问题覆盖率和延迟。

完整 180 条端到端回归：

- DIRECT、EXACT Fact Coverage 不低于 M9；
- 行为准确率不低于 78.89%；
- REFUSE 不低于 17/18；
- 有效 CLARIFY 保持 6/6；
- 明确记录逐题改善和退化；
- 若目标未达到，根据结果归因继续调整，不为追求指标硬编码评测问题。
