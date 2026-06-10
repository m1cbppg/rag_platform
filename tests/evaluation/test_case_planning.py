import json
from collections import Counter, defaultdict
from pathlib import Path

from src.rag_platform.evaluation.case_models import (
    CaseQuotaPlan,
    NoAnswerSubtype,
)
from src.rag_platform.evaluation.case_planner import (
    CaseSeedPlanner,
    load_case_context,
)
from src.rag_platform.evaluation.models import DatasetSplit, EvalCaseType


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _plan() -> CaseQuotaPlan:
    return CaseQuotaPlan.model_validate(
        {
            "case_type_counts": {
                "DIRECT": 90,
                "PARAPHRASE": 45,
                "EXACT": 30,
                "MULTI_CONDITION": 45,
                "MULTI_HOP": 30,
                "CONFLICT": 20,
                "NO_ANSWER": 40,
            },
            "split_case_type_counts": {
                "DEVELOPMENT": {
                    "DIRECT": 54,
                    "PARAPHRASE": 27,
                    "EXACT": 18,
                    "MULTI_CONDITION": 27,
                    "MULTI_HOP": 18,
                    "CONFLICT": 12,
                    "NO_ANSWER": 24,
                },
                "VALIDATION": {
                    "DIRECT": 18,
                    "PARAPHRASE": 9,
                    "EXACT": 6,
                    "MULTI_CONDITION": 9,
                    "MULTI_HOP": 6,
                    "CONFLICT": 4,
                    "NO_ANSWER": 8,
                },
                "TEST": {
                    "DIRECT": 18,
                    "PARAPHRASE": 9,
                    "EXACT": 6,
                    "MULTI_CONDITION": 9,
                    "MULTI_HOP": 6,
                    "CONFLICT": 4,
                    "NO_ANSWER": 8,
                },
            },
            "no_answer_subtype_counts": {
                "KNOWLEDGE_GAP": 20,
                "MISSING_CONDITION": 10,
                "OUT_OF_DOMAIN": 10,
            },
            "split_no_answer_subtype_counts": {
                "DEVELOPMENT": {
                    "KNOWLEDGE_GAP": 12,
                    "MISSING_CONDITION": 6,
                    "OUT_OF_DOMAIN": 6,
                },
                "VALIDATION": {
                    "KNOWLEDGE_GAP": 4,
                    "MISSING_CONDITION": 2,
                    "OUT_OF_DOMAIN": 2,
                },
                "TEST": {
                    "KNOWLEDGE_GAP": 4,
                    "MISSING_CONDITION": 2,
                    "OUT_OF_DOMAIN": 2,
                },
            },
            "split_topics": {
                "DEVELOPMENT": [
                    "order",
                    "refund",
                    "after_sales",
                    "payment",
                    "risk",
                ],
                "VALIDATION": ["coupon", "invoice", "member"],
                "TEST": ["logistics", "return"],
            },
        }
    )


def test_case_quota_plan_totals_are_consistent() -> None:
    plan = _plan()

    assert plan.total_case_count == 300
    assert plan.split_totals == {
        "DEVELOPMENT": 180,
        "VALIDATION": 60,
        "TEST": 60,
    }


def test_seed_planner_builds_exact_quotas_and_split_isolation() -> None:
    context = load_case_context(
        catalog_path=PROJECT_ROOT / "evaluation/corpus/catalog.json",
        document_blueprint_path=(
            PROJECT_ROOT
            / "evaluation/blueprints/ecommerce_document_plan.json"
        ),
    )
    seeds = CaseSeedPlanner(_plan(), context).build_seeds()

    assert len(seeds) == 300
    assert Counter(seed.case_type.value for seed in seeds) == {
        "DIRECT": 90,
        "PARAPHRASE": 45,
        "EXACT": 30,
        "MULTI_CONDITION": 45,
        "MULTI_HOP": 30,
        "CONFLICT": 20,
        "NO_ANSWER": 40,
    }
    assert Counter(seed.dataset_split.value for seed in seeds) == {
        "DEVELOPMENT": 180,
        "VALIDATION": 60,
        "TEST": 60,
    }
    no_answer = Counter(
        seed.no_answer_subtype
        for seed in seeds
        if seed.no_answer_subtype is not None
    )
    assert no_answer == {
        NoAnswerSubtype.KNOWLEDGE_GAP: 20,
        NoAnswerSubtype.MISSING_CONDITION: 10,
        NoAnswerSubtype.OUT_OF_DOMAIN: 10,
    }

    topic_splits = defaultdict(set)
    for seed in seeds:
        for topic in seed.source_topics:
            topic_splits[topic].add(seed.dataset_split.value)
    assert all(len(splits) == 1 for splits in topic_splits.values())


def test_seed_contracts_match_case_types() -> None:
    context = load_case_context(
        catalog_path=PROJECT_ROOT / "evaluation/corpus/catalog.json",
        document_blueprint_path=(
            PROJECT_ROOT
            / "evaluation/blueprints/ecommerce_document_plan.json"
        ),
    )
    seeds = CaseSeedPlanner(_plan(), context).build_seeds()

    for seed in seeds:
        if seed.case_type.value in {"DIRECT", "PARAPHRASE", "EXACT"}:
            assert len(seed.facts) == 1
        if seed.case_type.value == "MULTI_CONDITION":
            assert len(seed.facts) >= 2
            assert len({fact.source_doc_code for fact in seed.facts}) == 1
        if seed.case_type.value == "MULTI_HOP":
            assert len({fact.fact_key for fact in seed.facts}) >= 2
            assert len({fact.source_doc_code for fact in seed.facts}) >= 2
        if seed.case_type.value == "CONFLICT":
            assert len(seed.source_doc_codes) == 2
            assert seed.version_group
        if seed.case_type.value == "EXACT":
            assert seed.required_identifier
        if seed.case_type.value == "NO_ANSWER":
            assert seed.facts == []
            assert seed.source_doc_codes == []


def test_pool_seed_planner_adds_unique_replacement_candidates() -> None:
    context = load_case_context(
        catalog_path=PROJECT_ROOT / "evaluation/corpus/catalog.json",
        document_blueprint_path=(
            PROJECT_ROOT
            / "evaluation/blueprints/ecommerce_document_plan.json"
        ),
    )
    seeds = CaseSeedPlanner(_plan(), context).build_pool_seeds()

    assert len(seeds) > 300
    assert len({seed.seed_code for seed in seeds}) == len(seeds)


def test_supplement_seeds_prefer_facts_unused_by_initial_pool() -> None:
    context = load_case_context(
        catalog_path=PROJECT_ROOT / "evaluation/corpus/catalog.json",
        document_blueprint_path=(
            PROJECT_ROOT
            / "evaluation/blueprints/ecommerce_document_plan.json"
        ),
    )
    planner = CaseSeedPlanner(_plan(), context)
    initial = planner.build_pool_seeds()
    initial_direct_facts = {
        (seed.facts[0].source_doc_code, seed.facts[0].fact_key)
        for seed in initial
        if seed.dataset_split == DatasetSplit.DEVELOPMENT
        and seed.case_type == EvalCaseType.DIRECT
    }

    supplements = CaseSeedPlanner(
        _plan(),
        context,
    ).build_supplement_seeds(
        split=DatasetSplit.DEVELOPMENT,
        case_type=EvalCaseType.DIRECT,
        count=20,
        round_no=1,
    )

    assert len(supplements) == 20
    assert len({seed.seed_code for seed in supplements}) == 20
    assert all(
        seed.seed_code.startswith("SEED_SUPPLEMENT_R1_DIRECT_")
        for seed in supplements
    )
    assert all(
        seed.dataset_split == DatasetSplit.DEVELOPMENT
        and seed.case_type == EvalCaseType.DIRECT
        for seed in supplements
    )
    supplement_facts = {
        (seed.facts[0].source_doc_code, seed.facts[0].fact_key)
        for seed in supplements
    }
    assert supplement_facts.isdisjoint(initial_direct_facts)


def test_checked_in_case_plan_matches_contract() -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "evaluation/blueprints/ecommerce_case_plan.json"
        ).read_text(encoding="utf-8")
    )

    assert CaseQuotaPlan.model_validate(payload).total_case_count == 300
