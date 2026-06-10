from src.rag_platform.infrastructure.elasticsearch_store import (
    build_chunk_search_filters,
)


def test_builds_terms_filter_for_parent_business_domain() -> None:
    filters = build_chunk_search_filters(
        doc_type="RULE",
        business_domain="ecommerce_after_sales",
    )

    assert filters == [
        {"term": {"status": "ACTIVE"}},
        {"term": {"chunk_type": "RULE"}},
        {
            "terms": {
                "business_domain": [
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
                ]
            }
        },
    ]


def test_omits_business_domain_filter_when_not_provided() -> None:
    assert build_chunk_search_filters(
        doc_type=None,
        business_domain=None,
    ) == [{"term": {"status": "ACTIVE"}}]
