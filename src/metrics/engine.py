from __future__ import annotations

from typing import Iterable, List

from src.metrics.base import MetricCalculator, MetricResult
from src.metrics.registry import get_default_calculators
from src.models import EvaluationRun


class MetricsEngine:
    def __init__(self, calculators: Iterable[MetricCalculator] | None = None) -> None:
        # if calculators is None, the engine will resolve defaults based on run.evaluator
        self._provided_calculators = list(calculators) if calculators is not None else None

    def compute(self, run: EvaluationRun) -> List[MetricResult]:
        results: list[MetricResult] = []
        if self._provided_calculators is not None:
            calculators = self._provided_calculators
        else:
            calculators = get_default_calculators(run.evaluator, run.strategy)
        for calc in calculators:
            out = calc.compute(run)
            if isinstance(out, list):
                results.extend(out)
            else:
                results.append(out)
        return results