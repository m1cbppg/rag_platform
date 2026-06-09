import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from src.rag_platform.core.exceptions import ModelResponseFormatError
from src.rag_platform.evaluation.corpus_models import (
    CorpusManifestEntry,
    DocumentBlueprint,
    DocumentReviewResult,
    GeneratedSourceDocument,
    ReviewedDocumentOutcome,
    ReviewHistoryItem,
)
from src.rag_platform.evaluation.corpus_validation import (
    validate_generated_document,
)


class JsonChatClient(Protocol):
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


class DocumentGenerationService:
    def __init__(
        self,
        client: JsonChatClient,
        prompt_template: str,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.max_attempts = max_attempts

    async def generate(
        self,
        blueprint: DocumentBlueprint,
        review_feedback: list[str] | None = None,
        previous_document: GeneratedSourceDocument | None = None,
    ) -> GeneratedSourceDocument:
        last_error: Exception | None = None
        feedback = review_feedback or []

        for attempt in range(1, self.max_attempts + 1):
            user_prompt = self.prompt_template.format(
                blueprint_json=_json_text(blueprint.model_dump(mode="json")),
                review_feedback=_json_text(feedback),
                previous_document_json=_json_text(
                    previous_document.model_dump(mode="json")
                    if previous_document
                    else {}
                ),
                attempt=attempt,
            )
            try:
                payload = await self.client.chat_json(
                    system_prompt="你是企业电商售后知识库文档编写器。",
                    user_prompt=user_prompt,
                    temperature=0.4,
                    max_tokens=8192,
                )
                document = GeneratedSourceDocument.model_validate(payload)
                errors = validate_generated_document(blueprint, document)
                if errors:
                    raise ValueError("；".join(errors))
                return document
            except (
                ModelResponseFormatError,
                ValidationError,
                ValueError,
            ) as exc:
                last_error = exc
                feedback = [*feedback, f"第{attempt}次输出校验失败：{exc}"]

        raise ValueError(
            f"{blueprint.source_doc_code} 生成失败，已尝试{self.max_attempts}次"
        ) from last_error


class CorpusFileStore:
    def __init__(self, source_dir: Path, manifest_path: Path) -> None:
        self.source_dir = source_dir
        self.manifest_path = manifest_path

    def document_path(self, source_doc_code: str) -> Path:
        return self.source_dir / f"{source_doc_code}.json"

    def should_generate(self, source_doc_code: str, force: bool) -> bool:
        return force or not self.document_path(source_doc_code).exists()

    def load_document(self, source_doc_code: str) -> GeneratedSourceDocument:
        payload = json.loads(
            self.document_path(source_doc_code).read_text(encoding="utf-8")
        )
        return GeneratedSourceDocument.model_validate(payload)

    def save_document(
        self,
        document: GeneratedSourceDocument,
        generation_round: int,
        metadata: dict[str, Any] | None = None,
    ) -> CorpusManifestEntry:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        content = _json_text(document.model_dump(mode="json")) + "\n"
        output_path = self.document_path(document.source_doc_code)
        output_path.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

        entry = CorpusManifestEntry(
            source_doc_code=document.source_doc_code,
            status="GENERATED",
            relative_path=output_path.as_posix(),
            sha256=digest,
            generation_round=generation_round,
            metadata=metadata or {},
        )
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(_json_line(entry.model_dump(mode="json")) + "\n")
        return entry


class DocumentReviewService:
    def __init__(
        self,
        reviewer: JsonChatClient,
        generator: DocumentGenerationService,
        prompt_template: str,
        max_regeneration_rounds: int = 2,
        max_review_attempts: int = 2,
    ) -> None:
        self.reviewer = reviewer
        self.generator = generator
        self.prompt_template = prompt_template
        self.max_regeneration_rounds = max_regeneration_rounds
        self.max_review_attempts = max_review_attempts

    async def review(
        self,
        blueprint: DocumentBlueprint,
        document: GeneratedSourceDocument,
        related_documents: list[GeneratedSourceDocument],
    ) -> ReviewedDocumentOutcome:
        current_document = document
        history: list[ReviewHistoryItem] = []

        for round_no in range(self.max_regeneration_rounds + 1):
            review = await self._review_once(
                blueprint=blueprint,
                document=current_document,
                related_documents=related_documents,
            )
            history.append(ReviewHistoryItem(round_no=round_no, review=review))
            if review.passed:
                return ReviewedDocumentOutcome(
                    document=current_document,
                    review=review,
                    history=history,
                )
            if round_no == self.max_regeneration_rounds:
                break
            current_document = await self.generator.generate(
                blueprint=blueprint,
                review_feedback=review.issues,
                previous_document=current_document,
            )

        return ReviewedDocumentOutcome(
            document=current_document,
            review=history[-1].review,
            history=history,
        )

    async def _review_once(
        self,
        blueprint: DocumentBlueprint,
        document: GeneratedSourceDocument,
        related_documents: list[GeneratedSourceDocument],
    ) -> DocumentReviewResult:
        user_prompt = self.prompt_template.format(
            blueprint_json=_json_text(blueprint.model_dump(mode="json")),
            document_json=_json_text(document.model_dump(mode="json")),
            related_documents_json=_json_text(
                [item.model_dump(mode="json") for item in related_documents]
            ),
        )
        last_error: Exception | None = None
        for _ in range(self.max_review_attempts):
            try:
                payload = await self.reviewer.chat_json(
                    system_prompt="你是独立的企业知识库文档质量评审员。",
                    user_prompt=user_prompt,
                    temperature=0,
                    max_tokens=4096,
                )
                review = DocumentReviewResult.model_validate(payload)
                if review.source_doc_code != blueprint.source_doc_code:
                    raise ValueError(
                        "Qwen 审核结果的 source_doc_code 与蓝图不一致"
                    )
                return review
            except (
                ModelResponseFormatError,
                ValidationError,
                ValueError,
            ) as exc:
                last_error = exc
        raise ValueError(
            f"{blueprint.source_doc_code} 的Qwen审核响应连续格式异常"
        ) from last_error


def _json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _json_line(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
