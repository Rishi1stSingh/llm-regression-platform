from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


Status = Literal["passed", "failed"]
ChangeType = Literal["unchanged_pass", "unchanged_fail", "regression", "improvement"]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    input: str
    expected: str


@dataclass
class CaseResult:
    case_id: str
    input: str
    expected: str
    actual: str
    passed: bool
    reason: str = "Exact label match"
    baseline_actual: str | None = None
    baseline_passed: bool | None = None
    change: ChangeType | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationRun:
    version: str
    evaluator: str
    cases: list[CaseResult]
    run_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    baseline_run_id: str | None = None
    baseline_score: float | None = None
    score: float = 0.0
    delta: float | None = None
    status: Status = "passed"
    regressions: int = 0
    improvements: int = 0

    def finalize_score(self) -> None:
        self.score = sum(case.passed for case in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def passed_cases(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def failed_cases(self) -> int:
        return len(self.cases) - self.passed_cases

    def summary(self) -> dict:
        return {
            "run_id": self.run_id, "version": self.version, "evaluator": self.evaluator,
            "created_at": self.created_at, "baseline_run_id": self.baseline_run_id,
            "baseline_score": self.baseline_score, "score": self.score, "delta": self.delta,
            "status": self.status, "total_cases": len(self.cases),
            "passed_cases": self.passed_cases, "failed_cases": self.failed_cases,
            "regressions": self.regressions, "improvements": self.improvements,
        }
