from __future__ import annotations

from src.metrics.base import MetricResult
from src.models import EvaluationRun


class ExecutionAccuracyCalculator:
    """Fraction of cases where the generated SQL executed successfully and produced the expected result."""

    name = "execution_accuracy"

    def compute(self, run: EvaluationRun) -> MetricResult:
        if not run.cases:
            return MetricResult(name=self.name, value=0.0)
        passed = sum(1 for case in run.cases if case.passed)
        return MetricResult(name=self.name, value=passed / len(run.cases))


class ResultSetAccuracyCalculator:
    """Fraction of cases where the executed result set matches the expected result set."""

    name = "result_set_accuracy"

    def compute(self, run: EvaluationRun) -> MetricResult:
        if not run.cases:
            return MetricResult(name=self.name, value=0.0)
        passed = sum(1 for case in run.cases if case.passed and "Result set" in case.reason)
        return MetricResult(name=self.name, value=passed / len(run.cases))