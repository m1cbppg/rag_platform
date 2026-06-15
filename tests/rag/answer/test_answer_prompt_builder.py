from src.rag_platform.rag.answer.answer_prompt_builder import (
    AnswerPromptBuilder,
)


def test_prompt_requires_itemized_answers_for_sub_queries() -> None:
    _, user_prompt = AnswerPromptBuilder().build(
        question="订单怎么改地址，同时售后单要补什么材料？",
        rewritten_question=None,
        context="## 子问题 SQ1\n[C1] 地址规则",
        citations=[{"citation_id": "C1"}],
        sub_queries=[
            {"sub_query_id": "SQ1", "question": "地址条件？"},
            {"sub_query_id": "SQ2", "question": "材料要求？"},
        ],
    )

    assert "子问题回答计划" in user_prompt
    assert "SQ1：地址条件？" in user_prompt
    assert "SQ2：材料要求？" in user_prompt
    assert "逐个回答所有子问题" in user_prompt
    assert "证据不足时单独说明" in user_prompt


def test_simple_prompt_does_not_add_sub_query_instructions() -> None:
    _, user_prompt = AnswerPromptBuilder().build(
        question="怎么退款？",
        rewritten_question="退款规则",
        context="[C1] 退款规则",
        citations=[{"citation_id": "C1"}],
    )

    assert "子问题回答计划" not in user_prompt
    assert "逐个回答所有子问题" not in user_prompt
