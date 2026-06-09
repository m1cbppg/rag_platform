from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from src.rag_platform.evaluation.models import SourceDocumentType


class CorpusModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class RequiredFact(CorpusModel):
    fact_key: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)


class DocumentBlueprint(CorpusModel):
    source_doc_code: str = Field(min_length=1, max_length=64)
    doc_type: SourceDocumentType
    title: str = Field(min_length=1, max_length=255)
    topic: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    effective_from: date | None = None
    effective_to: date | None = None
    is_current: bool = True
    version_group: str | None = Field(default=None, max_length=100)
    supersedes: str | None = Field(default=None, max_length=64)
    conflicts_with: list[str] = Field(default_factory=list)
    required_facts: list[RequiredFact] = Field(min_length=1)
    required_identifiers: list[str] = Field(min_length=1)
    required_sections: list[str] = Field(min_length=1)
    generation_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_blueprint(self) -> "DocumentBlueprint":
        fact_keys = [item.fact_key for item in self.required_facts]
        if len(fact_keys) != len(set(fact_keys)):
            raise ValueError("required_facts 中的 fact_key 不能重复")
        if len(self.required_identifiers) != len(set(self.required_identifiers)):
            raise ValueError("required_identifiers 不能重复")
        if self.effective_from and self.effective_to:
            if self.effective_to < self.effective_from:
                raise ValueError("effective_to 不能早于 effective_from")
        if self.doc_type == SourceDocumentType.RULE and not self.effective_from:
            raise ValueError("RULE 蓝图必须提供 effective_from")
        return self


class GeneratedFact(CorpusModel):
    fact_key: str = Field(min_length=1, max_length=100)
    fact_text: str = Field(min_length=1)


class GeneratedDocumentSection(CorpusModel):
    section_code: str = Field(min_length=1, max_length=100)
    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    facts: list[GeneratedFact] = Field(default_factory=list)


class GeneratedSourceDocument(CorpusModel):
    source_doc_code: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    doc_type: SourceDocumentType
    topic: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    effective_from: date | None = None
    effective_to: date | None = None
    sections: list[GeneratedDocumentSection] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_document(self) -> "GeneratedSourceDocument":
        section_codes = [item.section_code for item in self.sections]
        if len(section_codes) != len(set(section_codes)):
            raise ValueError("section_code 不能重复")

        fact_keys = [
            fact.fact_key
            for section in self.sections
            for fact in section.facts
        ]
        if len(fact_keys) != len(set(fact_keys)):
            raise ValueError("文档中的 fact_key 不能重复")

        if self.effective_from and self.effective_to:
            if self.effective_to < self.effective_from:
                raise ValueError("effective_to 不能早于 effective_from")
        return self

    def plain_text(self) -> str:
        parts = [
            self.source_doc_code,
            self.title,
            self.doc_type.value,
            self.topic,
            self.version,
        ]
        for section in self.sections:
            parts.extend(
                [
                    section.section_code,
                    section.heading,
                    section.content,
                    *section.aliases,
                ]
            )
            for fact in section.facts:
                parts.extend([fact.fact_key, fact.fact_text])
        return "\n".join(parts)

    def fact_keys(self) -> set[str]:
        return {
            fact.fact_key
            for section in self.sections
            for fact in section.facts
        }

    def fact_texts(self) -> list[str]:
        return [
            fact.fact_text
            for section in self.sections
            for fact in section.facts
        ]


class DocumentReviewResult(CorpusModel):
    source_doc_code: str = Field(min_length=1, max_length=64)
    internal_consistency: float = Field(ge=0, le=1)
    fact_coverage: float = Field(ge=0, le=1)
    identifier_accuracy: float = Field(ge=0, le=1)
    structure_score: float = Field(ge=0, le=1)
    version_consistency: float = Field(ge=0, le=1)
    ambiguity_risk: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    issues: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)

    @computed_field
    @property
    def passed(self) -> bool:
        return (
            self.fact_coverage >= 1.0
            and self.identifier_accuracy >= 1.0
            and self.internal_consistency >= 0.90
            and self.structure_score >= 0.85
            and self.overall_score >= 0.88
        )


class ReviewHistoryItem(CorpusModel):
    round_no: int = Field(ge=0)
    review: DocumentReviewResult


class ReviewedDocumentOutcome(CorpusModel):
    document: GeneratedSourceDocument
    review: DocumentReviewResult
    history: list[ReviewHistoryItem]


class CorpusManifestEntry(CorpusModel):
    source_doc_code: str
    status: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_round: int = Field(ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
