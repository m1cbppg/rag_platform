# M9 自适应检索设计

## 目标

在不改变冻结评测集的前提下，把当前固定单轮检索升级为最多两轮的自适应检索。
系统必须能够判断当前证据是否值得直接进入回答阶段，并在证据不足时选择有原因、
可追踪的纠错策略。

本阶段只解决检索证据质量问题，不实现多跳问题拆解和 Claim 级答案校验。

## 当前问题

V0 和 M8 定向评测已经证明以下问题仍然存在：

1. 地址修改类问题可能完全召回不到标准证据。
2. 精确编号问题即使 Rerank 分数较高，也可能没有命中问题中的编号。
3. 新旧规则比较问题可能只保留一侧版本。
4. 多事实问题的必要证据可能在融合或精排阶段被淘汰。
5. 当前质量判断只检查候选数量，无法识别上述问题。
6. 评测追踪把所有结果记录为第 1 轮，无法解释重试带来的收益和成本。

## 设计原则

1. 最多执行两轮检索，控制延迟和模型费用。
2. 质量特征和质量策略分离，便于独立测试和阈值校准。
3. 确定性信号优先于 LLM，例如精确编号覆盖、版本数量和候选数量。
4. Query Rewrite 失败时必须有确定性兜底，不能让检索工作流失败。
5. 每轮查询、过滤条件、候选结果、质量特征和触发原因必须可观测。
6. 开发集用于校准，验证集用于选择配置，测试集只做最终一次评测。

## 质量特征

新增 `RetrievalQualityFeatures`，包含：

- `candidate_count`：多轮融合后的候选 Chunk 数。
- `distinct_document_count`：候选来自多少篇不同文档。
- `channel_overlap_at_10`：BM25 和 Vector 在前 10 名中的重合程度。
- `rerank_top1`：精排第一名分数。
- `rerank_top3_mean`：精排前三名平均分。
- `rerank_margin`：精排第一名与第二名分差。
- `target_type_coverage`：目标文档类型在候选中的覆盖比例。
- `exact_term_coverage`：问题中的编号、错误码等精确词在候选中的覆盖比例。
- `distinct_version_count`：候选中不同规则版本数量。
- `comparison_intent`：问题是否要求比较新旧规则或处理规则冲突。

精确词覆盖和版本覆盖用于修复单纯依赖 Rerank 分数无法发现的失败。

## 质量决策

质量策略输出：

```text
GOOD / WEAK / POOR
质量分
重试策略
触发原因
```

初始策略如下：

1. 没有候选：`POOR + RELAX_FILTER`。
2. 精确词未覆盖：`WEAK + FORCE_BM25`。
3. 比较问题只有一个版本：`WEAK + QUERY_REWRITE`。
4. 目标文档类型完全未覆盖：`WEAK + RELAX_FILTER`。
5. Rerank Top1 低于阈值且前三名均值偏低：`WEAK + QUERY_REWRITE`。
6. 其他情况：`GOOD`。

分差不能单独触发重试，因为多个分数接近的候选也可能分别支持不同必要事实。

## 重试策略

### QUERY_REWRITE

把原问题、当前查询、候选标题、候选短摘要和质量不足原因发送给 DeepSeek。
模型只负责输出检索表达，不负责回答问题。最多输出 3 条查询。

### FORCE_BM25

适用于错误码、规则编号、按钮名和字段名。保留精确词，并将第二轮路由切换为
BM25，避免语义候选挤掉精确匹配结果。

### RELAX_FILTER

优先移除单一文档类型过滤，但保留业务域限制。只有候选完全为空且没有文档类型
可放宽时，才允许移除业务域过滤。每次放宽的字段必须记录在尝试信息中。

## 多轮融合

每轮结果按 Chunk ID 去重，并使用加权 RRF 融合：

- 初始轮权重：1.0
- Query Rewrite：0.9
- Force BM25：1.0
- Relax Filter：0.8

融合结果保留：

- 命中的检索轮次
- Query 变体
- 检索通道
- 各轮原始排名和分数
- 多轮融合分数

第二轮完成后，对多轮融合候选重新执行一次 Rerank，最终精排结果进入 Context。

## LangGraph 流程

```text
analyze_query
-> retrieve
-> merge_rounds
-> rerank
-> judge_quality
   -> GOOD: build_context
   -> WEAK/POOR 且未达到最大轮次: prepare_retry
   -> 已达到最大轮次且有候选: build_context
   -> 已达到最大轮次且无候选: finish
-> finish
```

## 可观测性

`RagState`、工作流响应和评测适配器新增：

- 当前检索轮次
- 最大检索轮次
- 每轮检索尝试
- 最终质量特征和质量决策
- 重试策略和原因

评测运行时，每轮候选分别写入 `rag_eval_retrieval_hit`，从而能够计算：

- 二次检索触发率
- 各策略触发次数
- 第一轮失败、第二轮命中的 Case 数
- 重试带来的 Recall 和事实覆盖率增量
- 重试带来的 p50/p95 延迟增量

## 失败处理

- Query Rewrite 模型失败：使用原问题、精确词和候选标题组成确定性查询。
- 第二轮检索失败：保留第一轮结果继续构建 Context。
- Rerank 失败：沿用现有 fail-open 机制。
- 达到最大轮次仍无候选：由回答动作决策输出 REFUSE。
- 达到最大轮次证据仍较弱：质量状态保留为 WEAK，交由回答动作决策判断。

## 验收标准

1. 已知地址修改召回失败 Case 至少命中一个标准证据。
2. 精确编号 Case 的精确词覆盖率提高。
3. 冲突问题能够同时保留新旧版本候选。
4. 开发集 Recall@5 或 MRR 高于 V0。
5. MULTI_HOP 事实覆盖率高于 V0 的 75%。
6. 输出二次检索触发率及 p95 延迟增幅。
7. CLARIFY 维持 6/6，REFUSE 不低于 17/18。
8. 配置快照完整记录自适应检索开关、阈值、最大轮数和策略版本。

## 评测集校正

定向评测发现 v1 的 EXACT 题存在系统性标注缺口：问题声明了
`required_identifier`，但部分 Gold Chunk 只包含业务结论，不包含编号本身。
这会把正确召回“编号解释 Chunk + 业务处理 Chunk”的系统错误计为零召回。

校正方式：

1. v1 保持冻结不变。
2. 创建 `rag_eval_ecommerce:v2`。
3. 保留全部 300 道问题、答案、数据划分和原 Gold 证据。
4. 对 identifier 不在原 Gold Chunk 的 EXACT 题，在原 source_doc_codes
   对应文档中确定性查找 identifier。
5. 新增 identifier 所在 Chunk，使用独立 fact_key，并增加 required_fact_count。
6. 所有修正写入 generation_metadata，输出修正审计报告和新 SHA256。
7. V0 与 V1 都必须在 v2 上重新执行，禁止把 v1 基线与 v2 优化结果直接比较。

## 成对检索专项评测

端到端小样本 A/B 发现，即使模型温度为 0，LLM Query Understanding
仍可能在两次运行中生成不同的改写和扩展查询。若直接比较两次完整问答运行，
无法区分指标变化究竟来自上游 Query 波动，还是来自自适应检索。

因此 M9 增加独立的成对检索专项评测：

1. 强制使用规则 Query Analyzer，固定每条题的初始 Query 计划。
2. 每条 Case 先执行关闭自适应检索的控制组，再执行开启自适应检索的实验组。
3. 不调用 Answer Generator、动作决策和 Qwen Judge，只评估检索阶段。
4. 两组使用相同数据集 SHA、检索索引、Embedding、Rerank 和 Context 配置。
5. 只使用 DEVELOPMENT 中 expected_action=ANSWER 的 Case 做阈值分析。
6. 输出 Recall@K、MRR、nDCG、Fact Coverage、触发率、策略分布、
   改善/退化 Case 数和额外延迟。
7. 保存逐题初始 Query 计划、各轮尝试、最终 Chunk 和指标差值，支持失败归因。

该专项评测回答“自适应检索本身是否有效”；完整端到端评测继续回答
“整个 RAG 产品最终是否变好”。两者不能混为同一个实验结论。
