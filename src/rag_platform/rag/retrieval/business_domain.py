from collections.abc import Sequence


BUSINESS_DOMAIN_ALIAS_VERSION = "v1"

_BUSINESS_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "ecommerce_after_sales": (
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
    ),
}


def business_domain_alias_snapshot() -> dict[str, list[str]]:
    return {
        alias: list(domains)
        for alias, domains in _BUSINESS_DOMAIN_ALIASES.items()
    }


def resolve_business_domains(
    business_domain: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if business_domain is None:
        return ()
    values = (
        [business_domain]
        if isinstance(business_domain, str)
        else list(business_domain)
    )
    resolved: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        resolved.extend(
            _BUSINESS_DOMAIN_ALIASES.get(
                normalized,
                (normalized,),
            )
        )
    return tuple(dict.fromkeys(resolved))
