import json

from src.metrics.base import MetricResult
from src.models import CaseResult, EvaluationRun
from src.storage import SQLiteRepository


def _make_run(version: str = "v1", *, metric_values=None):
    cases = [
        CaseResult("1", "input one", "positive", "positive", True, "exact match"),
        CaseResult("2", "input two", "negative", "positive", False, "expected negative"),
    ]
    run = EvaluationRun(version, "classification", cases)
    run.finalize_score()
    if metric_values is None:
        metric_values = [
            MetricResult("accuracy", 0.5, {"total": 2, "correct": 1}),
            MetricResult("precision", 0.5, {"tp": 1, "fp": 1}),
            MetricResult("recall", 0.5, {"tp": 1, "fn": 1}),
            MetricResult("f1", 0.5, {"tp": 1, "fp": 1, "fn": 1}),
        ]
    run.metrics = metric_values
    return run


def test_save_run_persists_metrics(tmp_path):
    repo = SQLiteRepository(tmp_path / "evaluations.db")
    run = _make_run()

    repo.save_run(run)

    rows = repo.connection.execute(
        "SELECT metric_name, metric_value, details FROM evaluation_metrics WHERE run_id = ? ORDER BY metric_order",
        (run.run_id,),
    ).fetchall()
    assert [row["metric_name"] for row in rows] == ["accuracy", "precision", "recall", "f1"]
    assert [row["metric_value"] for row in rows] == [0.5, 0.5, 0.5, 0.5]
    assert json.loads(rows[0]["details"]) == {"total": 2, "correct": 1}


def test_load_run_with_metrics(tmp_path):
    repo = SQLiteRepository(tmp_path / "evaluations.db")
    run = _make_run(version="vmetrics")

    repo.save_run(run)
    loaded = repo.latest_successful_run("vmetrics")

    assert loaded is not None
    assert [m.name for m in loaded.metrics] == ["accuracy", "precision", "recall", "f1"]
    assert loaded.metrics[0].value == 0.5
    assert loaded.metrics[0].details == {"total": 2, "correct": 1}


def test_load_run_without_metrics_legacy_row(tmp_path):
    db_path = tmp_path / "legacy.db"
    repo = SQLiteRepository(db_path)
    legacy_run = _make_run(version="vlegacy")
    repo.connection.execute(
        "DELETE FROM evaluation_metrics WHERE run_id = ?",
        (legacy_run.run_id,),
    )
    repo.connection.execute(
        "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            legacy_run.run_id,
            legacy_run.version,
            legacy_run.evaluator,
            legacy_run.created_at,
            legacy_run.baseline_run_id,
            legacy_run.baseline_score,
            legacy_run.score,
            legacy_run.delta,
            legacy_run.status,
            legacy_run.regressions,
            legacy_run.improvements,
            len(legacy_run.cases),
            None,
        ),
    )
    repo.connection.executemany(
        "INSERT INTO evaluation_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                legacy_run.run_id,
                case.case_id,
                case.input,
                case.expected,
                case.actual,
                int(case.passed),
                case.reason,
                None,
            )
            for case in legacy_run.cases
        ],
    )
    repo.connection.commit()

    loaded = repo.latest_successful_run("vlegacy")

    assert loaded is not None
    assert loaded.metrics == []


def test_multiple_metrics_and_details_round_trip(tmp_path):
    repo = SQLiteRepository(tmp_path / "multi.db")
    run = _make_run(version="vmulti")
    run.metrics = [
        MetricResult("accuracy", 0.9667, {"correct": 29, "total": 30}),
        MetricResult("precision", 0.9, {"tp": 18, "fp": 2}),
    ]

    repo.save_run(run)
    loaded = repo.latest_successful_run("vmulti")

    assert loaded is not None
    assert len(loaded.metrics) == 2
    assert loaded.metrics[0].details == {"correct": 29, "total": 30}
    assert loaded.metrics[1].details == {"tp": 18, "fp": 2}
