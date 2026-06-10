from src.rag_platform.rag.retrieval.business_domain import (
    resolve_business_domains,
)


def test_resolves_ecommerce_parent_domain_to_searchable_subdomains() -> None:
    assert resolve_business_domains("ecommerce_after_sales") == (
        "order",
        "payment",
        "refund",
        "return",
        "after_sales",
        "logistics",
        "coupon",
        "invoice",
        "member",
        "risk",
    )


def test_keeps_specific_domain_and_handles_empty_value() -> None:
    assert resolve_business_domains("refund") == ("refund",)
    assert resolve_business_domains(" refund ") == ("refund",)
    assert resolve_business_domains(None) == ()
    assert resolve_business_domains("") == ()


def test_deduplicates_explicit_domain_sequence() -> None:
    assert resolve_business_domains(
        ["refund", "order", "refund"]
    ) == ("refund", "order")
