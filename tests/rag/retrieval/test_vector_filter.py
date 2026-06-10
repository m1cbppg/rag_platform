from src.rag_platform.rag.retrieval.vector_retriever import (
    build_milvus_filter_expression,
)


def test_builds_in_expression_for_parent_business_domain() -> None:
    expression = build_milvus_filter_expression(
        doc_type="RULE",
        business_domain="ecommerce_after_sales",
    )

    assert expression == (
        'status == "ACTIVE" and doc_type == "RULE" and '
        'business_domain in ["order", "payment", "refund", "return", '
        '"after_sales", "logistics", "coupon", "invoice", "member", "risk"]'
    )


def test_builds_exact_expression_for_specific_domain() -> None:
    assert build_milvus_filter_expression(
        doc_type=None,
        business_domain="refund",
    ) == 'status == "ACTIVE" and business_domain == "refund"'


def test_escapes_filter_values() -> None:
    assert build_milvus_filter_expression(
        doc_type='RULE"TEST',
        business_domain=None,
    ) == 'status == "ACTIVE" and doc_type == "RULE\\"TEST"'
