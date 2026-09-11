from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.models import EvaluationRun


@dataclass
class MetricResult:
    name: str
    value: float
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MetricCalculator(Protocol):
    def compute(self, run: EvaluationRun) -> MetricResult | list[MetricResult]:
        ...
