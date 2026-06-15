from dataclasses import dataclass
import re


EVIDENCE_CONSTRAINT_GUARD_VERSION = "v1"

_CONSTRAINT_PATTERN = re.compile(
    r"(?P<number>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+)"
    r"\s*(?P<unit>分钟|小时|个?工作日|天|日|元|%)"
)
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


@dataclass(frozen=True)
class EvidenceConstraintGap:
    missing_constraints: list[str]


class EvidenceConstraintService:
    def find_gap(
        self,
        *,
        question: str,
        context: str,
    ) -> EvidenceConstraintGap | None:
        question_constraints = _extract_constraints(question)
        if not question_constraints:
            return None
        context_constraints = set(_extract_constraints(context))
        missing = [
            constraint
            for constraint in question_constraints
            if constraint not in context_constraints
        ]
        if not missing:
            return None
        return EvidenceConstraintGap(
            missing_constraints=list(dict.fromkeys(missing))
        )


def _extract_constraints(value: str) -> list[str]:
    return [
        _canonical_constraint(
            match.group("number"),
            match.group("unit"),
        )
        for match in _CONSTRAINT_PATTERN.finditer(value)
    ]


def _canonical_constraint(number: str, unit: str) -> str:
    normalized_unit = unit.removeprefix("个")
    normalized_number = (
        number
        if number[0].isdigit()
        else str(_chinese_number(number))
    )
    return f"{normalized_number}{normalized_unit}"


def _chinese_number(value: str) -> int:
    if value == "十":
        return 10
    if "百" in value:
        hundreds, rest = value.split("百", maxsplit=1)
        base = _CHINESE_DIGITS.get(hundreds, 1) * 100
        return base + (_chinese_number(rest) if rest else 0)
    if "十" in value:
        tens, ones = value.split("十", maxsplit=1)
        return (
            _CHINESE_DIGITS.get(tens, 1) * 10
            + _CHINESE_DIGITS.get(ones, 0)
        )
    result = 0
    for char in value:
        result = result * 10 + _CHINESE_DIGITS[char]
    return result
