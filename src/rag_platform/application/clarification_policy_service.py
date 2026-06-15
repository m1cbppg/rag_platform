import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "config" / "clarification_policies.json"
)
CLARIFICATION_POLICY_VERSION = "v5"


class ClarificationSlotPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    values: list[str] = Field(min_length=1)


class ClarificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    trigger_groups: list[list[str]] = Field(min_length=1)
    required_any_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    required_slots: list[ClarificationSlotPolicy] = Field(min_length=1)
    clarification_question: str = Field(min_length=1)


class ClarificationPolicyRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    policies: list[ClarificationPolicy] = Field(min_length=1)


@dataclass(frozen=True)
class ClarificationPolicyMatch:
    policy_code: str
    confidence: float
    reason: str
    clarification_question: str
    missing_slot_codes: list[str]
    missing_slot_labels: list[str]


class ClarificationPolicyService:
    def __init__(
        self,
        registry: ClarificationPolicyRegistry | None = None,
    ) -> None:
        self.registry = registry or load_clarification_policy_registry()

    def detect(
        self,
        *,
        question: str,
    ) -> ClarificationPolicyMatch | None:
        normalized = _normalize(question)
        for policy in self.registry.policies:
            if not _matches_intent(normalized, policy.trigger_groups):
                continue
            if any(
                _normalize(term) in normalized
                for term in policy.excluded_terms
            ):
                continue
            if policy.required_any_terms and not any(
                _normalize(term) in normalized
                for term in policy.required_any_terms
            ):
                continue
            missing_slots = [
                slot
                for slot in policy.required_slots
                if not any(
                    _normalize(value) in normalized
                    for value in slot.values
                )
            ]
            if not missing_slots:
                return None
            labels = [slot.label for slot in missing_slots]
            return ClarificationPolicyMatch(
                policy_code=policy.code,
                confidence=0.99,
                reason=(
                    f"命中澄清策略{policy.code}，缺少必要条件："
                    + "、".join(labels)
                ),
                clarification_question=policy.clarification_question,
                missing_slot_codes=[
                    slot.code for slot in missing_slots
                ],
                missing_slot_labels=labels,
            )
        return None


@lru_cache(maxsize=1)
def load_clarification_policy_registry(
    path: Path = DEFAULT_POLICY_PATH,
) -> ClarificationPolicyRegistry:
    registry = ClarificationPolicyRegistry.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if registry.version != CLARIFICATION_POLICY_VERSION:
        raise ValueError(
            "澄清策略配置版本与代码版本不一致："
            f"{registry.version}!={CLARIFICATION_POLICY_VERSION}"
        )
    return registry


def clarification_policy_snapshot() -> dict:
    return json.loads(
        load_clarification_policy_registry().model_dump_json()
    )


def _matches_intent(
    normalized_question: str,
    trigger_groups: list[list[str]],
) -> bool:
    return all(
        any(
            _normalize(term) in normalized_question
            for term in group
        )
        for group in trigger_groups
    )


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()
