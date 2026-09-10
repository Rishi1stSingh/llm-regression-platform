from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
                improvements INTEGER NOT NULL, total_cases INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluation_cases (
                run_id TEXT NOT NULL, case_id TEXT NOT NULL, input TEXT NOT NULL,
                expected TEXT NOT NULL, actual TEXT NOT NULL, passed INTEGER NOT NULL,
                reason TEXT NOT NULL, PRIMARY KEY(run_id, case_id),
                FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
            );
        """)
        self.connection.commit()

    def save_run(self, run: EvaluationRun) -> None:
        self.connection.execute("""INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run.run_id, run.version, run.evaluator, run.created_at, run.baseline_run_id,
                  run.baseline_score, run.score, run.delta, run.status, run.regressions,
                  run.improvements, len(run.cases)))
        self.connection.executemany("""INSERT INTO evaluation_cases VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(run.run_id, c.case_id, c.input, c.expected, c.actual, int(c.passed), c.reason)
                   for c in run.cases])
        self.connection.commit()

    def latest_successful_run(self, version: str) -> EvaluationRun | None:
        row = self.connection.execute("""SELECT * FROM evaluation_runs WHERE version = ? AND status = 'passed'
            ORDER BY created_at DESC LIMIT 1""", (version,)).fetchone()
        return self._load_run(row) if row else None

    def _load_run(self, row: sqlite3.Row) -> EvaluationRun:
        case_rows = self.connection.execute("SELECT * FROM evaluation_cases WHERE run_id = ? ORDER BY case_id", (row["run_id"],)).fetchall()
        cases = [CaseResult(r["case_id"], r["input"], r["expected"], r["actual"], bool(r["passed"]), r["reason"])
                 for r in case_rows]
        return EvaluationRun(run_id=row["run_id"], version=row["version"], evaluator=row["evaluator"],
            created_at=row["created_at"], baseline_run_id=row["baseline_run_id"], baseline_score=row["baseline_score"],
            score=row["score"], delta=row["delta"], status=row["status"], regressions=row["regressions"],
            improvements=row["improvements"], cases=cases)
