import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.rag_platform.core.config import get_settings
from src.rag_platform.infrastructure.deepseek import DeepSeekClient
from src.rag_platform.rag.adaptive.models import (
    DecomposedSubQuery,
    QueryDecompositionResult,
)
from src.rag_platform.rag.adaptive.query_decomposition_prompt import (
    QUERY_DECOMPOSITION_SYSTEM_PROMPT,
    QUERY_DECOMPOSITION_USER_PROMPT,
)
from src.rag_platform.rag.adaptive.quality_features import (
    has_comparison_intent,
)


_ALLOWED_DOC_TYPES = {"FAQ", "SOP", "RULE", "MANUAL"}
_STRONG_COMPLEX_MARKERS = (
    "以及",
    "并且",
    "同时",
    "分别",
    "另外",
    "然后",
    "在此之前",
    "还需要",
)
_CHOICE_MARKERS = ("还是", "或者应该")


class _RawSubQuery(BaseModel):
    question: str = Field(min_length=2)
    target_doc_types: list[str] = Field(default_factory=list)
    depends_on_sub_query_id: str | None = None

    @field_validator("target_doc_types", mode="before")
    @classmethod
    def normalize_target_doc_types(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("depends_on_sub_query_id", mode="before")
    @classmethod
    def normalize_dependency(cls, value):
        if value is None:
            return None
        normalized = str(value).strip().upper()
        if normalized in {"", "NONE", "NULL"}:
            return None
        match = re.fullmatch(
            r"(?:SQ|SQ_|SUB_QUERY_?)?(\d+)",
            normalized,
        )
        return f"SQ{match.group(1)}" if match else normalized


class _RawDecomposition(BaseModel):
    requires_decomposition: bool
    decomposition_type: Literal["NONE", "PARALLEL", "DEPENDENT"]
    benefit_score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    sub_queries: list[_RawSubQuery] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_decomposition_type(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        raw_type = str(
            normalized.get("decomposition_type") or ""
        ).strip().upper()
        aliases = {
            "": "NONE",
            "NO": "NONE",
            "INDEPENDENT": "PARALLEL",
            "SEQUENTIAL": "DEPENDENT",
            "SERIAL": "DEPENDENT",
            "CHAIN": "DEPENDENT",
            "顺序": "DEPENDENT",
            "顺序依赖": "DEPENDENT",
        }
        normalized["decomposition_type"] = aliases.get(
            raw_type,
            raw_type,
        )
        return normalized


class QueryDecomposer:
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

    async def decompose(
        self,
        *,
        question: str,
        rewritten_question: str | None,
        target_doc_types: list[str],
        need_clarification: bool = False,
    ) -> QueryDecompositionResult:
        if not getattr(
            self.settings,
            "query_decomposition_enabled",
            True,
        ):
            return self._no_decomposition("查询分解未启用")
        if need_clarification:
            return self._no_decomposition(
                "Query分析要求先澄清，跳过查询分解"
            )
        if not self._is_decomposition_candidate(question):
            return self._no_decomposition("问题未命中复杂查询候选规则")

        max_sub_queries = int(
            getattr(
                self.settings,
                "query_decomposition_max_sub_queries",
                3,
            )
        )
        max_attempts = int(
            getattr(
                self.settings,
                "query_decomposition_max_attempts",
                2,
            )
        )
        model = str(
            getattr(
                self.settings,
                "query_decomposition_model",
                "deepseek-chat",
            )
        )
        client = self.client or self.client_factory()
        owns_client = self.client is None
        last_error: Exception | None = None
        try:
            prompt = QUERY_DECOMPOSITION_USER_PROMPT.format(
                question=question,
                rewritten_question=rewritten_question or question,
                target_doc_types=json.dumps(
                    target_doc_types,
                    ensure_ascii=False,
                ),
            )
            system_prompt = (
                QUERY_DECOMPOSITION_SYSTEM_PROMPT.format(
                    max_sub_queries=max_sub_queries
                )
            )
            for _ in range(max_attempts):
                try:
                    raw = await client.chat_json(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        model=model,
                        temperature=0,
                        max_tokens=768,
                    )
                    return self._parse_result(raw)
                except Exception as exc:
                    last_error = exc
            return self._fallback(last_error)
        finally:
            close = getattr(client, "aclose", None)
            if owns_client and close is not None:
                await close()

    def _parse_result(
        self,
        raw: dict,
    ) -> QueryDecompositionResult:
        parsed = _RawDecomposition.model_validate(raw)
        if not parsed.requires_decomposition:
            return self._no_decomposition(
                parsed.reason or "模型判断无需分解",
                decomposition_type=parsed.decomposition_type,
                benefit_score=parsed.benefit_score,
            )
        if (
            parsed.benefit_score
            < float(
                getattr(
                    self.settings,
                    "query_decomposition_min_benefit_score",
                    0.8,
                )
            )
        ):
            return self._no_decomposition(
                parsed.reason or "查询分解收益不足",
                decomposition_type=parsed.decomposition_type,
                benefit_score=parsed.benefit_score,
            )
        if (
            parsed.decomposition_type == "DEPENDENT"
            and (
                not bool(
                    getattr(
                        self.settings,
                        "query_decomposition_allow_dependent",
                        False,
                    )
                )
                or not bool(
                    getattr(
                        self.settings,
                        "dependent_multi_hop_enabled",
                        True,
                    )
                )
            )
        ):
            return self._no_decomposition(
                parsed.reason or "顺序依赖多跳未启用",
                decomposition_type=parsed.decomposition_type,
                benefit_score=parsed.benefit_score,
            )
        if parsed.decomposition_type not in {"PARALLEL", "DEPENDENT"}:
            return self._no_decomposition(
                parsed.reason or "模型未返回可执行的分解类型",
                decomposition_type=parsed.decomposition_type,
                benefit_score=parsed.benefit_score,
            )

        deduplicated: list[DecomposedSubQuery] = []
        seen: set[str] = set()
        for item in parsed.sub_queries:
            question = re.sub(r"\s+", " ", item.question).strip()
            normalized = question.rstrip("？?。 ").lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            doc_types = list(
                dict.fromkeys(
                    value.strip().upper()
                    for value in item.target_doc_types
                    if value.strip().upper() in _ALLOWED_DOC_TYPES
                )
            )
            deduplicated.append(
                DecomposedSubQuery(
                    sub_query_id=f"SQ{len(deduplicated) + 1}",
                    question=question,
                    target_doc_types=doc_types,
                    depends_on_sub_query_id=(
                        item.depends_on_sub_query_id
                    ),
                    is_template=(
                        "{{intermediate_fact}}" in question
                    ),
                )
            )
            if (
                len(deduplicated)
                >= int(
                    getattr(
                        self.settings,
                        "query_decomposition_max_sub_queries",
                        3,
                    )
                )
            ):
                break

        if len(deduplicated) < 2:
            raise ValueError("查询分解结果至少需要两个不同子问题")
        if parsed.decomposition_type == "DEPENDENT":
            if len(deduplicated) != 2:
                raise ValueError("顺序依赖查询固定需要两个子问题")
            first, second = deduplicated
            if first.depends_on_sub_query_id is not None:
                raise ValueError("第一跳不能依赖其他子问题")
            if second.depends_on_sub_query_id != first.sub_query_id:
                raise ValueError("第二跳必须依赖SQ1")
            if not second.is_template:
                raise ValueError(
                    "第二跳查询必须包含{{intermediate_fact}}占位符"
                )
        return QueryDecompositionResult(
            requires_decomposition=True,
            sub_queries=deduplicated,
            decomposition_type=parsed.decomposition_type,
            benefit_score=parsed.benefit_score,
            reason=parsed.reason.strip(),
            fallback_used=False,
        )

    def _is_decomposition_candidate(self, question: str) -> bool:
        normalized = question.strip()
        if (
            len(normalized)
            < int(
                getattr(
                    self.settings,
                    "query_decomposition_min_query_length",
                    18,
                )
            )
        ):
            return False
        if has_comparison_intent(normalized):
            return False
        if any(marker in normalized for marker in _CHOICE_MARKERS):
            return False
        question_count = normalized.count("?") + normalized.count("？")
        if question_count >= 2:
            return True
        return any(
            marker in normalized
            for marker in _STRONG_COMPLEX_MARKERS
        )

    @staticmethod
    def _no_decomposition(
        reason: str,
        *,
        decomposition_type: str = "NONE",
        benefit_score: float = 0.0,
    ) -> QueryDecompositionResult:
        return QueryDecompositionResult(
            requires_decomposition=False,
            sub_queries=[],
            decomposition_type=decomposition_type,
            benefit_score=benefit_score,
            reason=reason,
            fallback_used=False,
        )

    @staticmethod
    def _fallback(
        error: Exception | None,
    ) -> QueryDecompositionResult:
        reason = "查询分解不可用，回退原问题"
        if error is not None:
            details = re.sub(
                r"\s+",
                " ",
                str(error),
            ).strip()
            reason = (
                f"{reason}；{type(error).__name__}"
                + (f"：{details[:240]}" if details else "")
            )
        return QueryDecompositionResult(
            requires_decomposition=False,
            sub_queries=[],
            decomposition_type="NONE",
            benefit_score=0.0,
            reason=reason,
            fallback_used=True,
        )
