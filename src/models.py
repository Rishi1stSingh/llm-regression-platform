from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from src.comparator import MetricDelta, MetricRegressionCheck
    from src.metrics.base import MetricResult


Status = Literal["passed", "failed"]
ChangeType = Literal["unchanged_pass", "unchanged_fail", "regression", "improvement"]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    input: str
    expected: str
    # Optional task-specific fields (all with defaults so existing classification cases keep working)
    expected_result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    details: dict[str, Any] | None = None

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
    metrics: list[MetricResult] = field(default_factory=list)
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    metric_regression_checks: list[MetricRegressionCheck] = field(default_factory=list)
    strategy: str | None = None
    elapsed_seconds: float | None = None

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