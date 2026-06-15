import json
import re

from pydantic import BaseModel, Field

from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.deepseek import DeepSeekClient
from src.rag_platform.rag.adaptive.intermediate_fact_prompt import (
    INTERMEDIATE_FACT_SYSTEM_PROMPT,
    INTERMEDIATE_FACT_USER_PROMPT,
)
from src.rag_platform.rag.adaptive.models import IntermediateFactResult


class _RawIntermediateFact(BaseModel):
    success: bool
    intermediate_fact: str = ""
    evidence_quote: str = ""
    supporting_chunk_id: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class IntermediateFactExtractor:
    def __init__(
        self,
        *,
        settings=None,
        client=None,
        client_factory=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client
        self.client_factory = client_factory or DeepSeekClient

    async def extract(
        self,
        *,
        question: str,
        first_hop_question: str,
        next_query_template: str,
        candidate_documents: list[dict],
    ) -> IntermediateFactResult:
        candidates = self._candidate_payload(candidate_documents)
        if not candidates:
            return self._failure("第一跳没有可用于抽取中间事实的候选证据")

        client = self.client or self.client_factory()
        owns_client = self.client is None
        last_error: Exception | None = None
        try:
            user_prompt = INTERMEDIATE_FACT_USER_PROMPT.format(
                question=question,
                first_hop_question=first_hop_question,
                next_query_template=next_query_template,
                candidate_documents=json.dumps(
                    candidates,
                    ensure_ascii=False,
                ),
            )
            for _ in range(
                int(
                    getattr(
                        self.settings,
                        "dependent_fact_max_attempts",
                        2,
                    )
                )
            ):
                try:
                    raw = await client.chat_json(
                        system_prompt=INTERMEDIATE_FACT_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        model=str(
                            getattr(
                                self.settings,
                                "dependent_fact_model",
                                "deepseek-chat",
                            )
                        ),
                        temperature=0,
                        max_tokens=512,
                    )
                    return self._validate_result(raw, candidates)
                except Exception as exc:
                    last_error = exc
                    user_prompt = (
                        f"{user_prompt}\n\n"
                        "上一次输出未通过校验："
                        f"{str(exc).strip() or type(exc).__name__}。"
                        "请重新抽取一个直接回答第一跳问题的状态、"
                        "类别、等级、错误含义或规则名称；"
                        "不要抽取第二跳答案或连接短语。"
                    )
            return self._failure(
                "中间事实抽取不可用"
                + (
                    f"；{type(last_error).__name__}"
                    if last_error is not None
                    else ""
                ),
                fallback_used=True,
            )
        finally:
            close = getattr(client, "aclose", None)
            if owns_client and close is not None:
                await close()

    def _validate_result(
        self,
        raw: dict,
        candidates: list[dict],
    ) -> IntermediateFactResult:
        parsed = _RawIntermediateFact.model_validate(raw)
        if not parsed.success:
            return self._failure(
                parsed.reason or "第一跳证据不足以确定中间事实"
            )
        fact = re.sub(
            r"\s+",
            " ",
            parsed.intermediate_fact,
        ).strip()
        quote = parsed.evidence_quote.strip()
        if not fact or not quote or parsed.supporting_chunk_id is None:
            raise ValueError("中间事实、证据原文和支持Chunk不能为空")
        if self._is_connector_phrase(fact):
            raise ValueError(
                "中间事实只是流程连接短语，没有回答第一跳问题"
            )
        minimum_confidence = float(
            getattr(
                self.settings,
                "dependent_fact_min_confidence",
                0.75,
            )
        )
        if parsed.confidence < minimum_confidence:
            return self._failure(
                (
                    "中间事实置信度不足："
                    f"{parsed.confidence:.4f}<{minimum_confidence:.4f}"
                )
            )
        candidate_by_id = {
            int(item["chunk_id"]): item for item in candidates
        }
        candidate = candidate_by_id.get(parsed.supporting_chunk_id)
        if candidate is None:
            raise ValueError("支持Chunk不属于第一跳候选")
        if self._normalize(quote) not in self._normalize(
            str(candidate["content"])
        ):
            raise ValueError("证据原文不属于支持Chunk正文")
        return IntermediateFactResult(
            success=True,
            intermediate_fact=fact,
            evidence_quote=quote,
            supporting_chunk_id=parsed.supporting_chunk_id,
            confidence=parsed.confidence,
            reason=parsed.reason.strip(),
        )

    def _candidate_payload(
        self,
        documents: list[dict],
    ) -> list[dict]:
        result = []
        limit = int(
            getattr(
                self.settings,
                "dependent_fact_max_candidates",
                5,
            )
        )
        for document in documents:
            chunk_id = document.get("chunk_id")
            content = str(
                document.get("page_content")
                or document.get("content")
                or ""
            ).strip()
            if chunk_id is None or not content:
                continue
            result.append(
                {
                    "chunk_id": int(chunk_id),
                    "title": str(document.get("title") or ""),
                    "source_section": str(
                        document.get("source_section") or ""
                    ),
                    "content": content[:2400],
                }
            )
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", value).strip()

    @staticmethod
    def _is_connector_phrase(value: str) -> bool:
        normalized = re.sub(r"\s+", "", value).strip("，。；：")
        return bool(
            re.fullmatch(
                r"(?:核实|确认|判断|检查|查询|判定).{0,12}"
                r"(?:后|之后)",
                normalized,
            )
        )

    @staticmethod
    def _failure(
        reason: str,
        *,
        fallback_used: bool = False,
    ) -> IntermediateFactResult:
        return IntermediateFactResult(
            success=False,
            reason=reason,
            fallback_used=fallback_used,
        )
