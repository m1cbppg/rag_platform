# M10.1 评测集标签复核记录

## CASE_NO_ANSWER_003

问题：

```text
如果商品在运输过程中损坏，可以申请哪几种赔偿？
```

当前标签：

```text
expected_action=REFUSE
```

实际检索到的知识库证据明确说明：

- 运输破损时可为客户提供退款方案；
- 运输破损时可为客户提供补发方案。

系统回答“退款或补发”与 Context 直接一致，并且引用能够支持结论。因此该 Case
不是知识缺口题，不应要求系统拒答。

建议修正为：

```text
expected_action=ANSWER
reference_answer=运输过程中损坏的商品可以根据责任判定申请退款或补发。
required_fact_count=2
```

建议必要事实：

```text
transport_damage_refund
transport_damage_reship
```

本次没有直接修改冻结数据集，避免在没有人工确认和重新冻结版本的情况下改变
基线。后续应创建新的数据集版本并重新计算 SHA256。

