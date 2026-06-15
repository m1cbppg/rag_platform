import argparse
import json
from pathlib import Path
import sys

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_platform.infrastructure.mysql import (  # noqa: E402
    create_mysql_engine,
)


def validate_calibration_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "test":
        raise ValueError("测试集禁止用于检索质量阈值校准")
    if normalized not in {"development", "validation"}:
        raise ValueError("校准分片只能是development或validation")
    return normalized


def select_rerank_thresholds(
    rows: list[dict],
    *,
    top1_candidates: list[float],
    top3_candidates: list[float],
) -> dict:
    usable = [
        row
        for row in rows
        if row.get("fact_coverage") is not None
        and row.get("top1") is not None
        and row.get("top3_mean") is not None
    ]
    if not usable:
        raise ValueError("没有可用于校准的检索结果")
    failure_count = sum(
        1
        for row in usable
        if float(row["fact_coverage"]) < 1.0
    )
    success_count = len(usable) - failure_count
    if failure_count == 0:
        raise ValueError("校准数据中没有事实覆盖失败Case")

    candidates = []
    for top1 in top1_candidates:
        for top3 in top3_candidates:
            failure_hits = 0
            false_triggers = 0
            trigger_count = 0
            for row in usable:
                triggered = (
                    float(row["top1"]) < top1
                    and float(row["top3_mean"]) < top3
                )
                if triggered:
                    trigger_count += 1
                failed = float(row["fact_coverage"]) < 1.0
                if failed and triggered:
                    failure_hits += 1
                if not failed and triggered:
                    false_triggers += 1
            failure_recall = failure_hits / failure_count
            false_trigger_rate = (
                false_triggers / success_count
                if success_count
                else 0.0
            )
            trigger_rate = trigger_count / len(usable)
            objective = (
                failure_recall
                - 0.35 * false_trigger_rate
                - 0.05 * trigger_rate
            )
            candidates.append(
                {
                    "top1_threshold": top1,
                    "top3_threshold": top3,
                    "failure_recall": round(
                        failure_recall,
                        6,
                    ),
                    "false_trigger_rate": round(
                        false_trigger_rate,
                        6,
                    ),
                    "trigger_rate": round(trigger_rate, 6),
                    "objective": round(objective, 6),
                    "case_count": len(usable),
                }
            )
    return max(
        candidates,
        key=lambda item: (
            item["objective"],
            -item["false_trigger_rate"],
            -item["trigger_rate"],
            -item["top1_threshold"],
            -item["top3_threshold"],
        ),
    )


def load_run_rows(
    *,
    run_code: str,
    split: str,
) -> list[dict]:
    engine = create_mysql_engine()
    sql = text(
        """
        SELECT
            cr.fact_coverage,
            MAX(
                CASE
                    WHEN h.channel = 'RERANK' AND h.rank_no = 1
                    THEN h.rerank_score
                END
            ) AS top1,
            AVG(
                CASE
                    WHEN h.channel = 'RERANK' AND h.rank_no <= 3
                    THEN h.rerank_score
                END
            ) AS top3_mean
        FROM rag_eval_run r
        JOIN rag_eval_case_result cr ON cr.run_id = r.id
        JOIN rag_eval_case c ON c.id = cr.case_id
        LEFT JOIN rag_eval_retrieval_hit h
          ON h.case_result_id = cr.id
        WHERE r.run_code = :run_code
          AND c.dataset_split = :split
          AND c.expected_action = 'ANSWER'
          AND cr.status = 'SUCCESS'
        GROUP BY cr.id, cr.fact_coverage
        """
    )
    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                sql,
                {
                    "run_code": run_code,
                    "split": split.upper(),
                },
            ).mappings()
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据已完成评测运行推荐自适应检索Rerank阈值",
    )
    parser.add_argument("--run-code", required=True)
    parser.add_argument(
        "--split",
        default="development",
        choices=["development", "validation", "test"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = validate_calibration_split(args.split)
    rows = load_run_rows(
        run_code=args.run_code,
        split=split,
    )
    result = select_rerank_thresholds(
        rows,
        top1_candidates=[
            round(value / 100, 2)
            for value in range(45, 76, 5)
        ],
        top3_candidates=[
            round(value / 100, 2)
            for value in range(40, 71, 5)
        ],
    )
    result.update(
        {
            "run_code": args.run_code,
            "split": split,
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
