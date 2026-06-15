import pytest

from scripts.calibrate_retrieval_quality import (
    select_rerank_thresholds,
    validate_calibration_split,
)


def test_selects_thresholds_that_detect_failures_without_overtriggering() -> None:
    rows = [
        {"top1": 0.45, "top3_mean": 0.42, "fact_coverage": 0.0},
        {"top1": 0.50, "top3_mean": 0.47, "fact_coverage": 0.5},
        {"top1": 0.72, "top3_mean": 0.66, "fact_coverage": 1.0},
        {"top1": 0.86, "top3_mean": 0.78, "fact_coverage": 1.0},
    ]

    result = select_rerank_thresholds(
        rows,
        top1_candidates=[0.50, 0.55, 0.60],
        top3_candidates=[0.45, 0.50, 0.55],
    )

    assert result["failure_recall"] == 1.0
    assert result["false_trigger_rate"] == 0.0
    assert result["top1_threshold"] in {0.55, 0.60}
    assert result["top3_threshold"] in {0.50, 0.55}


def test_calibration_rejects_test_split() -> None:
    validate_calibration_split("development")
    validate_calibration_split("validation")

    with pytest.raises(ValueError, match="测试集"):
        validate_calibration_split("test")
