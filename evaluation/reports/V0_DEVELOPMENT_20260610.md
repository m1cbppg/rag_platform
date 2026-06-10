# V0_DEVELOPMENT_20260610 RAG 基线评测报告

## 运行信息

- 实验版本：`V0`
- 实验名称：`baseline-hybrid-rrf-rerank`
- 运行状态：`SUCCESS`
- 总题数：180
- 归因通过率：10.00%
- Judge 通过率：10.00%
- 行为准确率：10.00%

## 系统级根因

### 严重：业务域精确过滤不匹配

评测题业务域为 `ecommerce_after_sales`，ACTIVE Chunk 业务域为 `after_sales, coupon, invoice, logistics, member, order, payment, refund, return, risk, 长FAQ, 长SOP`，两者没有交集。当前 BM25 和向量检索都使用业务域精确过滤，因此候选结果会在检索阶段被全部过滤。

处理建议：统一业务域枚举，或把顶层业务域映射为允许的细分域集合；修复后创建新的实验版本重跑，不覆盖本次 V0 基线。

## 检索漏斗

| 阶段 | 必要事实平均覆盖率 |
| --- | ---: |
| 融合召回 | 0.00% |
| 精排结果 | 0.00% |
| 最终 Context | 0.00% |

## 失败归因分布

| 归因 | 数量 | 占比 | 优化建议 |
| --- | ---: | ---: | --- |
| 融合召回完全缺失 | 156 | 86.67% | 优先检查 Query 改写、检索路由、过滤条件、索引内容和召回 Top K。 |
| 通过 | 18 | 10.00% | 保持当前链路，后续用于回归对照。 |
| 系统缺少澄清行为 | 6 | 3.33% | 把 Query 分析中的 need_clarification 接入 Chat 行为输出。 |

## 题型表现

| 题型 | 题数 | 归因通过率 | Judge通过率 | Recall@10 | Fact Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| CONFLICT | 12 | 0.00% | 0.00% | 0.0000 | 0.0000 |
| DIRECT | 54 | 0.00% | 0.00% | 0.0000 | 0.0000 |
| EXACT | 18 | 0.00% | 0.00% | 0.0000 | 0.0000 |
| MULTI_CONDITION | 27 | 0.00% | 0.00% | 0.0000 | 0.0000 |
| MULTI_HOP | 18 | 0.00% | 0.00% | 0.0000 | 0.0000 |
| NO_ANSWER | 24 | 75.00% | 75.00% | - | - |
| PARAPHRASE | 27 | 0.00% | 0.00% | 0.0000 | 0.0000 |

## 难度表现

| 难度 | 题数 | 归因通过率 | Judge通过率 | Recall@10 | Fact Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| EASY | 113 | 13.27% | 13.27% | 0.0000 | 0.0000 |
| HARD | 8 | 12.50% | 12.50% | 0.0000 | 0.0000 |
| MEDIUM | 59 | 3.39% | 3.39% | 0.0000 | 0.0000 |

## M8 优化优先级

1. **融合召回完全缺失**：156 题，占 86.67%。优先检查 Query 改写、检索路由、过滤条件、索引内容和召回 Top K。
2. **系统缺少澄清行为**：6 题，占 3.33%。把 Query 分析中的 need_clarification 接入 Chat 行为输出。

## 典型失败案例

| Case | 题型 | 难度 | 问题 | 预期/实际 | 主因 | Recall@10 | Fact Coverage |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| CASE_CONFLICT_001 | CONFLICT | EASY | 普通商品的退款时限，在新旧规则下有什么区别？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_002 | CONFLICT | MEDIUM | 根据新旧订单取消规则，已发货的订单到底能不能取消？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_003 | CONFLICT | MEDIUM | 预售商品和虚拟商品在订单取消方面，新旧规则有什么不同规定？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_004 | CONFLICT | MEDIUM | 订单取消规则的不同版本中，对于哪些特殊商品类型有豁免或限制？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_005 | CONFLICT | MEDIUM | 订单取消流程在新旧规则下有什么不同？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_006 | CONFLICT | MEDIUM | 售后退款时限的不同版本中，对于哪些商品类别有特殊的例外规定？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_007 | CONFLICT | HARD | 如果我的订单已发货但未签收，并且使用了优惠券，申请退款时新旧规则分别怎么处理？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_008 | CONFLICT | HARD | 如果定制商品出现质量问题，是否还能享受无理由退货？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_009 | CONFLICT | MEDIUM | 普通商品签收后多久可以申请无理由退款？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_010 | CONFLICT | MEDIUM | 订单取消规则从V1更新到V2后，对于取消条件和流程有哪些关键变化？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_011 | CONFLICT | HARD | 订单已经发货了还能取消吗？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_CONFLICT_012 | CONFLICT | MEDIUM | 在订单取消规则中，V1和V2对于已发货或已出库订单的处理方式有何不同？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_001 | DIRECT | EASY | 哪种类型的商品不适用两小时取消窗口？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_002 | DIRECT | EASY | 原路退款自动重试最多几次？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_003 | DIRECT | EASY | 处理仅退款申请时，第一步需要做什么？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_004 | DIRECT | EASY | 预售商品是否适用两小时取消窗口？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_005 | DIRECT | EASY | 在什么条件下可以修改未出库订单的地址？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_006 | DIRECT | EASY | 商品有质量问题，退款有时间限制吗？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_007 | DIRECT | EASY | 订单被风控核查期间，退款或补偿操作会怎样？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_008 | DIRECT | EASY | 对于已发货但未签收的订单，申请退款前需要做什么？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_009 | DIRECT | EASY | 我下了一个订单还没付款，能自己取消吗？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_010 | DIRECT | EASY | 支付处理中状态超过30分钟怎么办？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_011 | DIRECT | EASY | 已发货的订单是否可以取消？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_012 | DIRECT | EASY | 在什么条件下可以修改订单的收货地址？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_013 | DIRECT | EASY | 订单已经发货但还没签收，我该怎么操作才能申请退款？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_014 | DIRECT | EASY | 原路退款重试失败后，应该怎么处理？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_015 | DIRECT | EASY | 在风险核查期间，系统会如何处理自动退款或补偿？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_016 | DIRECT | EASY | 支付遇到问题，第一步该做什么？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_017 | DIRECT | EASY | V2版本的订单取消规则中，取消窗口如何判断？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |
| CASE_DIRECT_018 | DIRECT | EASY | 在风控核查期间，对于自动退款或补偿操作有什么要求？ | ANSWER/REFUSE | 融合召回完全缺失 | 0.0000 | 0.0000 |

## 说明

- 运行状态 SUCCESS 表示评测流程完成，不表示答案质量通过。
- 主归因采用上游优先原则，避免把召回问题重复计算为答案问题。
- JSON 报告包含全部逐题证据、阶段覆盖率和次级归因。
