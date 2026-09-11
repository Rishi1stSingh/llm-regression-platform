from __future__ import annotations

from src.metrics.base import MetricResult
from src.models import EvaluationRun
from src.evaluators.classification import ClassificationEvaluator


class AccuracyCalculator:
    name = "accuracy"

    def compute(self, run: EvaluationRun) -> MetricResult:
        total = len(run.cases)
        if total == 0:
            return MetricResult(name=self.name, value=0.0)
        passed = sum(1 for case in run.cases if case.passed)
        value = passed / total
        return MetricResult(name=self.name, value=value)


def _labels() -> list[str]:
    # reuse canonical labels from ClassificationEvaluator for consistency
    return sorted(list(ClassificationEvaluator.labels))


class PrecisionCalculator:
    name = "precision"

    def compute(self, run: EvaluationRun) -> MetricResult:
        labels = _labels()
        if not run.cases:
            return MetricResult(name=self.name, value=0.0, details={l: 0.0 for l in labels})
        per_class: dict[str, float] = {}
        for label in labels:
            tp = sum(1 for c in run.cases if c.expected == label and c.actual == label)
            fp = sum(1 for c in run.cases if c.expected != label and c.actual == label)
            denom = tp + fp
            per_class[label] = (tp / denom) if denom > 0 else 0.0
        macro = sum(per_class.values()) / len(labels)
        return MetricResult(name=self.name, value=macro, details=per_class)


class RecallCalculator:
    name = "recall"

    def compute(self, run: EvaluationRun) -> MetricResult:
        labels = _labels()
        if not run.cases:
            return MetricResult(name=self.name, value=0.0, details={l: 0.0 for l in labels})
        per_class: dict[str, float] = {}
        for label in labels:
            tp = sum(1 for c in run.cases if c.expected == label and c.actual == label)
            fn = sum(1 for c in run.cases if c.expected == label and c.actual != label)
            denom = tp + fn
            per_class[label] = (tp / denom) if denom > 0 else 0.0
        macro = sum(per_class.values()) / len(labels)
        return MetricResult(name=self.name, value=macro, details=per_class)


class F1Calculator:
    name = "f1"

    def compute(self, run: EvaluationRun) -> MetricResult:
        labels = _labels()
        if not run.cases:
            return MetricResult(name=self.name, value=0.0, details={l: 0.0 for l in labels})
        precisions: dict[str, float] = {}
        recalls: dict[str, float] = {}
        for label in labels:
            tp = sum(1 for c in run.cases if c.expected == label and c.actual == label)
            fp = sum(1 for c in run.cases if c.expected != label and c.actual == label)
            fn = sum(1 for c in run.cases if c.expected == label and c.actual != label)
            p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precisions[label] = p
            recalls[label] = r
        per_class: dict[str, float] = {}
        for label in labels:
            p = precisions[label]
            r = recalls[label]
            per_class[label] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        macro = sum(per_class.values()) / len(labels)
        return MetricResult(name=self.name, value=macro, details=per_class)
