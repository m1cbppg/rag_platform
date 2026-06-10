# M7 V0 业务域过滤修复设计

## 目标

修复顶层业务域 `ecommerce_after_sales` 与 Chunk 细分业务域不一致导致的全量零召回问题，并重新运行一份可以真实衡量召回、精排、Context 和答案质量的正式 V0 基线。

## 已确认根因

评测题统一携带顶层业务域：

```text
ecommerce_after_sales
```

Chunk 使用细分业务域：

```text
order、payment、refund、return、after_sales、logistics、
coupon、invoice、member、risk
```

BM25 使用 Elasticsearch `term` 精确过滤，向量检索使用 Milvus `==` 精确过滤。顶层域与细分域没有相等值，因此两个召回通道都返回空结果。

Gold Chunk 已确认全部存在于 Elasticsearch 和 Milvus，本次问题不是索引缺失。

## 修复方案

增加集中式业务域解析器：

```text
ecommerce_after_sales
    -> order
    -> payment
    -> refund
    -> return
    -> after_sales
    -> logistics
    -> coupon
    -> invoice
    -> member
    -> risk
```

规则：

1. 顶层业务域展开为多个细分域；
2. 已经是细分域时仍保持单值精确过滤；
3. 未提供业务域时不添加业务域过滤；
4. Elasticsearch 多值过滤使用 `terms`；
5. Milvus 多值过滤使用 `in`；
6. BM25 和向量检索必须使用同一个解析器；
7. M7 报告的系统级诊断必须使用解析后的业务域判断是否冲突。

## 不采用的方案

### 评测脚本传入空业务域

这会绕过问题，但生产 Chat 仍会在收到顶层业务域时零召回，属于评测特判。

### 修改冻结评测集业务域

冻结数据集不能为适配当前实现而修改，并且单道题可能跨越多个细分业务域。

### 完全删除业务域过滤

会扩大所有请求的检索范围，破坏现有细分域隔离能力。

## 验证流程

1. 单元测试验证业务域展开；
2. 单元测试验证 ES `terms` 和 Milvus `in`；
3. 使用一条真实 DEVELOPMENT 题验证 Gold Chunk 能进入检索结果；
4. 使用 5 条题做小批量评测；
5. 使用新 Run Code 完整运行 180 条题；
6. 实验版本保持 `V0`；
7. 生成正式中文报告并确认召回漏斗不再全部为 0。

## 正式运行标识

- 实验版本：`V0`
- 实验名称：`baseline-hybrid-rrf-rerank-domain-fixed`
- Run Code：`V0_DEVELOPMENT_BASELINE_20260610`

旧 Run `V0_DEVELOPMENT_20260610` 保留为无效故障运行，不作为正式质量基线。
