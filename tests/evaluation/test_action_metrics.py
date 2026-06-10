import pytest

from src.rag_platform.evaluation.action_metrics import (
    action_correct,
    evaluate_actions,
)
from src.rag_platform.evaluation.models import ActualAction, ExpectedAction


def test_action_correct_requires_exact_behavior_match() -> None:
    assert action_correct(ExpectedAction.ANSWER, ActualAction.ANSWER) is True
    assert action_correct(ExpectedAction.REFUSE, ActualAction.REFUSE) is True
    assert action_correct(ExpectedAction.CLARIFY, ActualAction.REFUSE) is False
    assert action_correct(ExpectedAction.ANSWER, ActualAction.ERROR) is False


def test_evaluate_actions_builds_three_by_four_confusion_matrix() -> None:
    result = evaluate_actions(
        expected_actions=[
            ExpectedAction.ANSWER,
            ExpectedAction.REFUSE,
            ExpectedAction.CLARIFY,
            ExpectedAction.ANSWER,
        ],
        actual_actions=[
            ActualAction.ANSWER,
            ActualAction.ANSWER,
            ActualAction.CLARIFY,
            ActualAction.ERROR,
        ],
    )

    assert result.total == 4
    assert result.correct == 2
    assert result.accuracy == 0.5
    assert result.confusion_matrix["ANSWER"] == {
        "ANSWER": 1,
        "REFUSE": 0,
        "CLARIFY": 0,
        "ERROR": 1,
    }
    assert result.confusion_matrix["REFUSE"]["ANSWER"] == 1
    assert result.confusion_matrix["CLARIFY"]["CLARIFY"] == 1


def test_evaluate_actions_calculates_refuse_and_clarify_scores() -> None:
    result = evaluate_actions(
        expected_actions=[
            ExpectedAction.REFUSE,
            ExpectedAction.REFUSE,
            ExpectedAction.CLARIFY,
            ExpectedAction.ANSWER,
        ],
        actual_actions=[
            ActualAction.REFUSE,
            ActualAction.ANSWER,
            ActualAction.CLARIFY,
            ActualAction.REFUSE,
        ],
    )

    refuse = result.per_action["REFUSE"]
    clarify = result.per_action["CLARIFY"]
    assert refuse.precision == 0.5
    assert refuse.recall == 0.5
    assert refuse.f1 == 0.5
    assert clarify.precision == 1.0
    assert clarify.recall == 1.0
    assert clarify.f1 == 1.0


def test_evaluate_actions_tracks_errors_as_a_separate_prediction() -> None:
    result = evaluate_actions(
        expected_actions=[ExpectedAction.ANSWER, ExpectedAction.REFUSE],
        actual_actions=[ActualAction.ERROR, ActualAction.ERROR],
    )

    assert result.error_count == 2
    assert result.per_action["ERROR"].predicted == 2
    assert result.per_action["ERROR"].true_positive == 0


def test_evaluate_actions_rejects_different_list_lengths() -> None:
    with pytest.raises(ValueError, match="数量必须一致"):
        evaluate_actions(
            expected_actions=[ExpectedAction.ANSWER],
            actual_actions=[],
        )


def test_evaluate_actions_returns_none_accuracy_for_empty_input() -> None:
    result = evaluate_actions(expected_actions=[], actual_actions=[])

    assert result.total == 0
    assert result.accuracy is None
