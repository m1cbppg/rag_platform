# M10 查询分解成对评测报告

## 实验范围

- 数据集：`rag_eval_ecommerce:v2`
- 数据划分：`DEVELOPMENT`
- Case 数：45
- 两组均开启 M9 自适应检索并固定规则 Query Analysis
- 控制组关闭查询分解，实验组开启查询分解

## 核心结果

| 指标 | M9 控制组 | M10 分解组 | 差值 |
|---|---:|---:|---:|
| Recall@1 | 0.5222 | 0.5222 | +0.0000 |
| Recall@3 | 0.7000 | 0.7111 | +0.0111 |
| Recall@5 | 0.7778 | 0.8000 | +0.0222 |
| Recall@10 | 0.8222 | 0.8556 | +0.0333 |
| MRR | 0.8204 | 0.8315 | +0.0111 |
| nDCG@5 | 0.7443 | 0.7563 | +0.0121 |
| nDCG@10 | 0.7631 | 0.7800 | +0.0169 |
| 事实覆盖率 | 0.8222 | 0.8556 | +0.0333 |

## 分解触发与成本

- 分解触发率：46.67% （21/45）
- 上游 Query 计划一致率：100.00%
- 平均子问题数：2.00
- 平均子问题覆盖率：100.00%
- 子问题完全覆盖 Case：21/21
- 平均延迟：3497.3 ms → 4542.5 ms （+1045.3 ms）
- P95 延迟：9100.0 ms → 9558.4 ms （+458.4 ms）

## 分类型结果

| 类型 | Case 数 | 触发数 | Fact Coverage 控制组 | 分解组 | 差值 |
|---|---:|---:|---:|---:|---:|
| MULTI_CONDITION | 27 | 16 | 0.8333 | 0.8889 | +0.0556 |
| MULTI_HOP | 18 | 5 | 0.8056 | 0.8056 | +0.0000 |

## Case 变化

- 事实覆盖率：改善 2，退化 0，不变 43。

| Case | 类型 | 是否分解 | 子问题覆盖率 | Fact Coverage 差值 | 延迟差值(ms) |
|---|---|---|---:|---:|---:|
| CASE_MULTI_CONDITION_019 | MULTI_CONDITION | 是 | 100.00% | +1.0000 | +4166 |
| CASE_MULTI_CONDITION_005 | MULTI_CONDITION | 是 | 100.00% | +0.5000 | +2736 |
| CASE_MULTI_CONDITION_001 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | -1487 |
| CASE_MULTI_CONDITION_002 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +7453 |
| CASE_MULTI_CONDITION_003 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | -935 |
| CASE_MULTI_CONDITION_004 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +1065 |
| CASE_MULTI_CONDITION_006 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +127 |
| CASE_MULTI_CONDITION_007 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +1540 |
| CASE_MULTI_CONDITION_008 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +18 |
| CASE_MULTI_CONDITION_009 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +5536 |
| CASE_MULTI_CONDITION_010 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +1426 |
| CASE_MULTI_CONDITION_011 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +9963 |
| CASE_MULTI_CONDITION_012 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +3157 |
| CASE_MULTI_CONDITION_013 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | -4747 |
| CASE_MULTI_CONDITION_014 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | -20 |
| CASE_MULTI_CONDITION_015 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +1793 |
| CASE_MULTI_CONDITION_016 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +2629 |
| CASE_MULTI_CONDITION_017 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +892 |
| CASE_MULTI_CONDITION_018 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +3144 |
| CASE_MULTI_CONDITION_020 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +228 |
| CASE_MULTI_CONDITION_021 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +1793 |
| CASE_MULTI_CONDITION_022 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +149 |
| CASE_MULTI_CONDITION_023 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | -4927 |
| CASE_MULTI_CONDITION_024 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | -9625 |
| CASE_MULTI_CONDITION_025 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +1519 |
| CASE_MULTI_CONDITION_026 | MULTI_CONDITION | 是 | 100.00% | +0.0000 | +6231 |
| CASE_MULTI_CONDITION_027 | MULTI_CONDITION | 否 | 100.00% | +0.0000 | +39 |
| CASE_MULTI_HOP_001 | MULTI_HOP | 是 | 100.00% | +0.0000 | +7801 |
| CASE_MULTI_HOP_002 | MULTI_HOP | 是 | 100.00% | +0.0000 | +2125 |
| CASE_MULTI_HOP_003 | MULTI_HOP | 否 | 100.00% | +0.0000 | +81 |
| CASE_MULTI_HOP_004 | MULTI_HOP | 是 | 100.00% | +0.0000 | +6112 |
| CASE_MULTI_HOP_005 | MULTI_HOP | 是 | 100.00% | +0.0000 | +3249 |
| CASE_MULTI_HOP_006 | MULTI_HOP | 否 | 100.00% | +0.0000 | +1808 |
| CASE_MULTI_HOP_007 | MULTI_HOP | 否 | 100.00% | +0.0000 | -4146 |
| CASE_MULTI_HOP_008 | MULTI_HOP | 否 | 100.00% | +0.0000 | +1392 |
| CASE_MULTI_HOP_009 | MULTI_HOP | 否 | 100.00% | +0.0000 | +81 |
| CASE_MULTI_HOP_010 | MULTI_HOP | 否 | 100.00% | +0.0000 | -1382 |
| CASE_MULTI_HOP_011 | MULTI_HOP | 是 | 100.00% | +0.0000 | +2943 |
| CASE_MULTI_HOP_012 | MULTI_HOP | 否 | 100.00% | +0.0000 | -4780 |
| CASE_MULTI_HOP_013 | MULTI_HOP | 否 | 100.00% | +0.0000 | +11 |
| CASE_MULTI_HOP_014 | MULTI_HOP | 否 | 100.00% | +0.0000 | +1403 |
| CASE_MULTI_HOP_015 | MULTI_HOP | 否 | 100.00% | +0.0000 | -2566 |
| CASE_MULTI_HOP_016 | MULTI_HOP | 否 | 100.00% | +0.0000 | -1033 |
| CASE_MULTI_HOP_017 | MULTI_HOP | 否 | 100.00% | +0.0000 | +81 |
| CASE_MULTI_HOP_018 | MULTI_HOP | 否 | 100.00% | +0.0000 | -6 |
