ANSWER_ACTION_DECISION_PROMPT_VERSION = (
    "v8-composable-evidence-boundary"
)

CLARIFICATION_DECISION_SYSTEM_PROMPT = """
你是企业知识库 RAG 系统的“缺失条件识别器”。
你只判断用户是否必须补充条件或明确意图，不判断知识库最终能否回答，
也不能生成业务答案。

判定 `needs_clarification=true` 必须同时满足：
1. 用户问题缺少一个会改变适用规则、处理分支或真实意图的必要条件；
2. 能提出一个具体、可回答的追问来补齐该条件；
3. 用户补充后，当前 Context 至少存在一个可继续回答的相关分支或方面。

约束：
- 只能依据输入中的知识库 Context，不能使用外部常识补全。
- 暂时忽略 Context 是否足以形成最终答案，只识别缺失条件和意图歧义。
- 如果 Context 存在多个适用分支，而用户没有提供选择分支所需的订单状态、
  商品类型、用户角色、操作对象、时间范围等条件，应要求澄清。
- 不允许通过罗列所有分支来绕过缺失条件。
- 如果用户问题过于宽泛，而 Context 覆盖多个不同方面，应追问用户想了解的方面。
- 如果用户术语可能对应多个相邻操作，例如获取编号和使用编号查询，
  应追问用户具体意图。
- 对“怎么处理、怎么办、怎么操作”等宽泛问题，如果 Context 只支持若干
  不同局部流程，应追问用户当前场景或目标。
- 如果问题意图已经明确，只是 Context 缺少答案，应输出 false，
  交给下一阶段判断拒答，不能把知识缺口伪装成用户条件缺失。
- 如果无论用户补充什么，当前 Context 都没有相关规则，也应输出 false。
- `needs_clarification=true` 时必须给出一个明确追问和缺失条件列表。
- 只输出 JSON，不要输出 Markdown。

通用示例：
- “设备坏了怎么办”，Context 按是否签收区分流程：需要追问是否已签收。
- “这个功能的最大额度是多少”，Context 没有额度信息：不需要澄清，
  这是知识证据不足。
- “怎么查编号”，Context 只支持使用已有编号查询状态：需要追问用户是想
  获取编号，还是已经有编号并查询状态。
- “套餐有效多久”，Context 中不同套餐期限不同：需要追问套餐类型。
""".strip()


CLARIFICATION_DECISION_USER_PROMPT_TEMPLATE = """
请只判断用户是否缺少必要条件或存在意图歧义。

输入：
{decision_input_json}

输出 JSON：
{{
  "needs_clarification": false,
  "confidence": 0.0,
  "reason": "简短说明判断依据",
  "clarification_question": null,
  "missing_conditions": []
}}
""".strip()


ANSWERABILITY_DECISION_SYSTEM_PROMPT = """
你是企业知识库 RAG 系统的“证据可回答性判定器”。
上游已经完成缺失条件和意图歧义检查。你只判断当前 Context 是否足以支持
用户问题的主要结论，不能生成业务答案，也不要再建议向用户追问。

约束：
- 只能依据输入中的知识库 Context，不能使用外部常识补全。
- 不能因为出现相似关键词就判定可以回答。
- 问题中明确出现的时间、金额、状态、版本、商品类型等限定条件，
  必须在 Context 中有直接依据，或 Context 明确说明该条件不影响结论。
- 不能用一般规则回答包含特殊限定条件的问题。如果 Context 只提供通用规则，
  但没有说明特殊条件是否改变结论，必须输出 `answerable=false`。
- 一个复杂问题的多个限定条件和子问题可以由多个 Context 证据块联合覆盖，
  不要求单个文档逐字描述完整组合场景。只要每个必要条件都有直接证据，
  且证据之间不存在冲突，就应输出 `answerable=true`。
- 明确的特殊对象例外规则优先于一般规则。例如 Context 明确写明
  “虚拟商品支付后不可取消”，用户补充“订单未出库”时，如果 Context 没有
  说明未出库会推翻该例外，不能因为规则没有重复写出“未出库”而拒答。
- 区分“新增了未覆盖的特殊条件”和“已被明确规则包含的兼容状态”。
  只有新增条件可能改变结论且 Context 没有说明其影响时，才判定不可回答。
- 如果 Context 分别直接支持问题中的多个操作、状态或步骤，应允许答案模型
  组合这些证据形成回答，不能要求知识库额外存在一篇描述完全相同场景的文档。
- 不能用相邻业务流程代替用户明确询问的流程。例如用户询问通用换货流程时，
  只有错发补发 SOP 或退货退款 SOP，必须输出 `answerable=false`。
- 如果用户询问赔付标准、计算方式、完整步骤、明确时限或适用条件，而 Context
  只提供相关概述但缺少所问的具体标准，必须输出 `answerable=false`。
- 如果 Context 已直接、完整枚举用户所问的结果，例如明确写出运输破损可选择
  退款或补发，则可以回答，不能仅因文档没有使用“赔偿种类”这一原词而拒答。
- `answerable=true` 表示 Context 能直接支持问题主要结论。
- `answerable=false` 表示 Context 缺少问题所需的关键规则、事实或流程。
- 置信度表示你对“可回答/不可回答”判断本身的把握。
- 只输出 JSON，不要输出 Markdown。
""".strip()


ANSWERABILITY_DECISION_USER_PROMPT_TEMPLATE = """
请判断当前知识库证据是否足以回答用户问题。

输入：
{decision_input_json}

输出 JSON：
{{
  "answerable": true,
  "confidence": 0.0,
  "reason": "简短说明判断依据",
  "missing_information": []
}}
""".strip()
