from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.metrics.base import MetricResult
from src.models import CaseResult, EvaluationRun


class SQLiteRepository:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY, version TEXT NOT NULL, evaluator TEXT NOT NULL,
                created_at TEXT NOT NULL, baseline_run_id TEXT, baseline_score REAL,
                score REAL NOT NULL, delta REAL, status TEXT NOT NULL, regressions INTEGER NOT NULL,
                improvements INTEGER NOT NULL, total_cases INTEGER NOT NULL, strategy TEXT
            );
            CREATE TABLE IF NOT EXISTS evaluation_cases (
                run_id TEXT NOT NULL, case_id TEXT NOT NULL, input TEXT NOT NULL,
                expected TEXT NOT NULL, actual TEXT NOT NULL, passed INTEGER NOT NULL,
                reason TEXT NOT NULL, details TEXT,
                PRIMARY KEY(run_id, case_id),
                FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                run_id TEXT NOT NULL, metric_order INTEGER NOT NULL, metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL, details TEXT,
                PRIMARY KEY(run_id, metric_order),
                FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
            );
        """)
        # Migrate legacy schemas: add new columns if they don't already exist
        self._add_column_if_missing("evaluation_runs", "strategy", "TEXT")
        self._add_column_if_missing("evaluation_cases", "details", "TEXT")
        self.connection.commit()

    def _add_column_if_missing(self, table: str, column: str, column_type: str) -> None:
        columns = [row[1] for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def save_run(self, run: EvaluationRun) -> None:
        self.connection.execute("""INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run.run_id, run.version, run.evaluator, run.created_at, run.baseline_run_id,
                  run.baseline_score, run.score, run.delta, run.status, run.regressions,
                  run.improvements, len(run.cases), run.strategy))
        self.connection.executemany("""INSERT INTO evaluation_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(run.run_id, c.case_id, c.input, c.expected, c.actual, int(c.passed), c.reason,
                   json.dumps(c.details, sort_keys=True) if c.details is not None else None)
                   for c in run.cases])
        self.connection.executemany(
            """INSERT INTO evaluation_metrics VALUES (?, ?, ?, ?, ?)""",
            [
                (run.run_id, index, metric.name, metric.value, json.dumps(metric.details, sort_keys=True) if metric.details is not None else None)
                for index, metric in enumerate(run.metrics)
            ],
        )
        self.connection.commit()

    def latest_successful_run(self, version: str, evaluator: str | None = None) -> EvaluationRun | None:
        """Latest successful run for a version, optionally restricted to an evaluator type.

        The evaluator filter is required by the regression gate: comparing a
        summarization/text_to_sql run against a classification baseline would
        mix incompatible scores, metrics, and case outputs. Callers that pass
        no evaluator keep the legacy behavior (any evaluator type).
        """
        if evaluator is not None:
            row = self.connection.execute(
                """SELECT * FROM evaluation_runs WHERE version = ? AND evaluator = ? AND status = 'passed'
                ORDER BY created_at DESC LIMIT 1""",
                (version, evaluator),
            ).fetchone()
        else:
            row = self.connection.execute(
                """SELECT * FROM evaluation_runs WHERE version = ? AND status = 'passed'
                ORDER BY created_at DESC LIMIT 1""",
                (version,),
            ).fetchone()
        return self._load_run(row) if row else None

    def _load_run(self, row: sqlite3.Row) -> EvaluationRun:
        case_rows = self.connection.execute("SELECT * FROM evaluation_cases WHERE run_id = ? ORDER BY case_id", (row["run_id"],)).fetchall()
        cases = []
        for r in case_rows:
            details = r["details"] if "details" in r.keys() else None
            cases.append(CaseResult(
                r["case_id"], r["input"], r["expected"], r["actual"], bool(r["passed"]), r["reason"],
                details=json.loads(details) if details is not None else None,
            ))
        metrics = self._load_metrics(row["run_id"])
        strategy = row["strategy"] if "strategy" in row.keys() else None
        return EvaluationRun(run_id=row["run_id"], version=row["version"], evaluator=row["evaluator"],
            created_at=row["created_at"], baseline_run_id=row["baseline_run_id"], baseline_score=row["baseline_score"],
            score=row["score"], delta=row["delta"], status=row["status"], regressions=row["regressions"],
            improvements=row["improvements"], cases=cases, metrics=metrics, strategy=strategy)

    def _load_metrics(self, run_id: str) -> list[MetricResult]:
        rows = self.connection.execute(
            "SELECT metric_name, metric_value, details FROM evaluation_metrics WHERE run_id = ? ORDER BY metric_order",
            (run_id,),
        ).fetchall()
        metrics: list[MetricResult] = []
        for row in rows:
            details = row["details"]
            metrics.append(
                MetricResult(
                    name=row["metric_name"],
                    value=row["metric_value"],
                    details=json.loads(details) if details is not None else None,
                )
            )
        return metrics