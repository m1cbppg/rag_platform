import json
import math
from collections import defaultdict
from pathlib import Path

from src.rag_platform.evaluation.case_models import (
    CaseQuotaPlan,
    CaseSeed,
    CaseSourceDocument,
    CaseSourceFact,
    NoAnswerSubtype,
)
from src.rag_platform.evaluation.models import (
    DatasetSplit,
    EvalCaseType,
    ExpectedAction,
)


def load_case_plan(path: Path) -> CaseQuotaPlan:
    return CaseQuotaPlan.model_validate_json(path.read_text(encoding="utf-8"))


def load_case_context(
    *,
    catalog_path: Path,
    document_blueprint_path: Path,
) -> dict[str, CaseSourceDocument]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    blueprints = json.loads(
        document_blueprint_path.read_text(encoding="utf-8")
    )
    blueprint_by_code = {
        item["source_doc_code"]: item for item in blueprints
    }
    documents: dict[str, CaseSourceDocument] = {}
    for item in catalog["documents"]:
        code = item["source_doc_code"]
        blueprint = blueprint_by_code[code]
        documents[code] = CaseSourceDocument(
            source_doc_code=code,
            mapped_doc_id=item["mapped_doc_id"],
            doc_type=item["doc_type"],
            title=item["title"],
            topic=item["topic"],
            version=item["version"],
            version_group=blueprint.get("version_group"),
            required_identifiers=blueprint["required_identifiers"],
            facts=[
                CaseSourceFact(
                    source_doc_code=code,
                    fact_key=fact["fact_key"],
                    fact_text=fact["fact_text"],
                    chunk_ids=fact["chunk_ids"],
                )
                for fact in item["facts"]
            ],
            chunks=item["chunks"],
        )
    return documents


class CaseSeedPlanner:
    def __init__(
        self,
        plan: CaseQuotaPlan,
        documents: dict[str, CaseSourceDocument],
    ) -> None:
        self.plan = plan
        self.documents = documents
        self._type_sequence: dict[EvalCaseType, int] = defaultdict(int)

    def build_seeds(self) -> list[CaseSeed]:
        seeds: list[CaseSeed] = []
        for split in (
            DatasetSplit.DEVELOPMENT,
            DatasetSplit.VALIDATION,
            DatasetSplit.TEST,
        ):
            documents = self._documents_for_split(split)
            for case_type, count in self.plan.split_case_type_counts[
                split
            ].items():
                if case_type == EvalCaseType.NO_ANSWER:
                    seeds.extend(self._build_no_answer_seeds(split))
                else:
                    seeds.extend(
                        self._build_answer_seeds(
                            split=split,
                            documents=documents,
                            case_type=case_type,
                            count=count,
                        )
                    )
        return seeds

    def build_pool_seeds(self) -> list[CaseSeed]:
        multiplier = self.plan.generation_pool_multiplier
        seeds: list[CaseSeed] = []
        for split in (
            DatasetSplit.DEVELOPMENT,
            DatasetSplit.VALIDATION,
            DatasetSplit.TEST,
        ):
            documents = self._documents_for_split(split)
            for case_type, count in self.plan.split_case_type_counts[
                split
            ].items():
                if case_type == EvalCaseType.NO_ANSWER:
                    subtype_counts = {
                        subtype: math.ceil(value * multiplier)
                        for subtype, value in (
                            self.plan.split_no_answer_subtype_counts[
                                split
                            ].items()
                        )
                    }
                    seeds.extend(
                        self._build_no_answer_seeds(
                            split,
                            subtype_counts=subtype_counts,
                        )
                    )
                else:
                    seeds.extend(
                        self._build_answer_seeds(
                            split=split,
                            documents=documents,
                            case_type=case_type,
                            count=math.ceil(count * multiplier),
                        )
                    )
        return seeds

    def build_supplement_seeds(
        self,
        *,
        split: DatasetSplit,
        case_type: EvalCaseType,
        count: int,
        round_no: int,
    ) -> list[CaseSeed]:
        if count < 1:
            raise ValueError("补充种子数量必须大于0")
        if round_no < 1:
            raise ValueError("补充轮次必须大于0")
        if case_type not in {
            EvalCaseType.DIRECT,
            EvalCaseType.PARAPHRASE,
            EvalCaseType.EXACT,
        }:
            raise ValueError(
                "当前定向补充仅支持DIRECT、PARAPHRASE和EXACT"
            )

        documents = self._documents_for_split(split)
        pool_count = math.ceil(
            self.plan.split_case_type_counts[split][case_type]
            * self.plan.generation_pool_multiplier
        )
        initial_fact_keys = set()
        for index in range(pool_count):
            document = documents[index % len(documents)]
            fact = document.facts[index % len(document.facts)]
            initial_fact_keys.add(
                (document.source_doc_code, fact.fact_key)
            )

        fact_units = [
            (document, fact)
            for document in documents
            for fact in document.facts
            if (
                case_type != EvalCaseType.EXACT
                or document.required_identifiers
            )
        ]
        unused_units = [
            unit
            for unit in fact_units
            if (
                unit[0].source_doc_code,
                unit[1].fact_key,
            )
            not in initial_fact_keys
        ]
        ordered_units = unused_units + fact_units
        offset = (round_no - 1) * count

        seeds = []
        for index in range(count):
            document, fact = ordered_units[
                (offset + index) % len(ordered_units)
            ]
            required_identifier = None
            if case_type == EvalCaseType.EXACT:
                required_identifier = document.required_identifiers[
                    (offset + index) % len(document.required_identifiers)
                ]
            seeds.append(
                self._seed(
                    split=split,
                    case_type=case_type,
                    documents=[document],
                    facts=[fact],
                    required_identifier=required_identifier,
                    seed_code=(
                        f"SEED_SUPPLEMENT_R{round_no}_"
                        f"{case_type.value}_{index + 1:03d}"
                    ),
                    variant_index=pool_count + offset + index + 1,
                )
            )
        return seeds

    def _documents_for_split(
        self,
        split: DatasetSplit,
    ) -> list[CaseSourceDocument]:
        allowed_topics = set(self.plan.split_topics[split])
        documents = sorted(
            (
                document
                for document in self.documents.values()
                if document.topic in allowed_topics
            ),
            key=lambda item: item.source_doc_code,
        )
        if not documents:
            raise ValueError(f"{split.value} 没有可用源文档")
        return documents

    def _build_answer_seeds(
        self,
        *,
        split: DatasetSplit,
        documents: list[CaseSourceDocument],
        case_type: EvalCaseType,
        count: int,
    ) -> list[CaseSeed]:
        if case_type == EvalCaseType.CONFLICT:
            return self._build_conflict_seeds(split, documents, count)

        seeds = []
        for index in range(count):
            if case_type == EvalCaseType.MULTI_HOP:
                first = documents[(index * 2) % len(documents)]
                second = documents[(index * 2 + 1) % len(documents)]
                if first.source_doc_code == second.source_doc_code:
                    second = documents[(index * 2 + 2) % len(documents)]
                selected_documents = [first, second]
                facts = [
                    first.facts[index % len(first.facts)],
                    second.facts[(index + 1) % len(second.facts)],
                ]
            else:
                document = documents[index % len(documents)]
                selected_documents = [document]
                if case_type == EvalCaseType.MULTI_CONDITION:
                    facts = [
                        document.facts[index % len(document.facts)],
                        document.facts[(index + 1) % len(document.facts)],
                    ]
                else:
                    facts = [document.facts[index % len(document.facts)]]

            required_identifier = None
            if case_type == EvalCaseType.EXACT:
                document = selected_documents[0]
                required_identifier = document.required_identifiers[
                    index % len(document.required_identifiers)
                ]
            seeds.append(
                self._seed(
                    split=split,
                    case_type=case_type,
                    documents=selected_documents,
                    facts=facts,
                    required_identifier=required_identifier,
                )
            )
        return seeds

    def _build_conflict_seeds(
        self,
        split: DatasetSplit,
        documents: list[CaseSourceDocument],
        count: int,
    ) -> list[CaseSeed]:
        groups: dict[str, list[CaseSourceDocument]] = defaultdict(list)
        for document in documents:
            if document.version_group:
                groups[document.version_group].append(document)
        pairs = [
            sorted(group, key=lambda item: item.version)
            for group in groups.values()
            if len(group) >= 2
        ]
        if not pairs:
            raise ValueError(f"{split.value} 缺少新旧版本冲突文档")

        seeds = []
        for index in range(count):
            pair = pairs[index % len(pairs)][:2]
            facts = [
                pair[0].facts[index % len(pair[0].facts)],
                pair[1].facts[index % len(pair[1].facts)],
            ]
            seeds.append(
                self._seed(
                    split=split,
                    case_type=EvalCaseType.CONFLICT,
                    documents=pair,
                    facts=facts,
                    version_group=pair[0].version_group,
                )
            )
        return seeds

    def _build_no_answer_seeds(
        self,
        split: DatasetSplit,
        subtype_counts: dict[NoAnswerSubtype, int] | None = None,
    ) -> list[CaseSeed]:
        seeds = []
        subtype_counts = (
            subtype_counts
            or self.plan.split_no_answer_subtype_counts[split]
        )
        for subtype, count in subtype_counts.items():
            action = (
                ExpectedAction.CLARIFY
                if subtype == NoAnswerSubtype.MISSING_CONDITION
                else ExpectedAction.REFUSE
            )
            for _ in range(count):
                seeds.append(
                    self._seed(
                        split=split,
                        case_type=EvalCaseType.NO_ANSWER,
                        documents=[],
                        facts=[],
                        expected_action=action,
                        no_answer_subtype=subtype,
                    )
                )
        return seeds

    def _seed(
        self,
        *,
        split: DatasetSplit,
        case_type: EvalCaseType,
        documents: list[CaseSourceDocument],
        facts: list[CaseSourceFact],
        expected_action: ExpectedAction = ExpectedAction.ANSWER,
        required_identifier: str | None = None,
        version_group: str | None = None,
        no_answer_subtype: NoAnswerSubtype | None = None,
        seed_code: str | None = None,
        variant_index: int | None = None,
    ) -> CaseSeed:
        self._type_sequence[case_type] += 1
        sequence = self._type_sequence[case_type]
        topics = sorted({document.topic for document in documents})
        source_group = (
            "+".join(f"topic:{topic}" for topic in topics)
            if topics
            else (
                f"no_answer:{split.value}:"
                f"{no_answer_subtype.value}"
            )
        )
        return CaseSeed(
            seed_code=(
                seed_code
                or f"SEED_{case_type.value}_{sequence:03d}"
            ),
            case_type=case_type,
            dataset_split=split,
            expected_action=expected_action,
            source_doc_codes=[
                document.source_doc_code for document in documents
            ],
            source_topics=topics,
            source_group=source_group,
            facts=facts,
            target_doc_types=sorted(
                {document.doc_type for document in documents},
                key=lambda item: item.value,
            ),
            required_identifier=required_identifier,
            version_group=version_group,
            no_answer_subtype=no_answer_subtype,
            variant_index=variant_index or sequence,
        )
