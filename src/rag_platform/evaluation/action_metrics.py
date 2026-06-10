from collections.abc import Iterable
from dataclasses import dataclass

from src.rag_platform.evaluation.models import ActualAction, ExpectedAction


_EXPECTED_ACTIONS = (
    ExpectedAction.ANSWER,
    ExpectedAction.REFUSE,
    ExpectedAction.CLARIFY,
)
_ACTUAL_ACTIONS = (
    ActualAction.ANSWER,
    ActualAction.REFUSE,
    ActualAction.CLARIFY,
    ActualAction.ERROR,
)


@dataclass(frozen=True)
class PerActionMetrics:
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int
    true_positive: int


@dataclass(frozen=True)
class ActionMetricsResult:
    confusion_matrix: dict[str, dict[str, int]]
    per_action: dict[str, PerActionMetrics]
    total: int
    correct: int
    error_count: int
    accuracy: float | None


def action_correct(
    expected_action: ExpectedAction | str,
    actual_action: ActualAction | str,
) -> bool:
    expected = ExpectedAction(expected_action)
    actual = ActualAction(actual_action)
    return expected.value == actual.value


def evaluate_actions(
    *,
    expected_actions: Iterable[ExpectedAction | str],
    actual_actions: Iterable[ActualAction | str],
) -> ActionMetricsResult:
    expected = [ExpectedAction(item) for item in expected_actions]
    actual = [ActualAction(item) for item in actual_actions]
    if len(expected) != len(actual):
        raise ValueError("预期行为和实际行为数量必须一致")

    matrix = {
        expected_action.value: {
            actual_action.value: 0
            for actual_action in _ACTUAL_ACTIONS
        }
        for expected_action in _EXPECTED_ACTIONS
    }
    for expected_action, actual_action in zip(expected, actual):
        matrix[expected_action.value][actual_action.value] += 1

    per_action = {}
    for actual_label in _ACTUAL_ACTIONS:
        label = actual_label.value
        support = sum(
            1
            for expected_action in expected
            if expected_action.value == label
        )
        predicted = sum(
            1
            for actual_action in actual
            if actual_action.value == label
        )
        true_positive = sum(
            1
            for expected_action, actual_action in zip(expected, actual)
            if expected_action.value == label
            and actual_action.value == label
        )
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, support)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_action[label] = PerActionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            predicted=predicted,
            true_positive=true_positive,
        )

    correct = sum(
        action_correct(expected_action, actual_action)
        for expected_action, actual_action in zip(expected, actual)
    )
    total = len(expected)
    return ActionMetricsResult(
        confusion_matrix=matrix,
        per_action=per_action,
        total=total,
        correct=correct,
        error_count=sum(
            item == ActualAction.ERROR for item in actual
        ),
        accuracy=(correct / total if total else None),
    )


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
