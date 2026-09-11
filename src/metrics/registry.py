from __future__ import annotations

from typing import Dict, Iterable, Type

from src.metrics.base import MetricCalculator
from src.metrics.classification import AccuracyCalculator, PrecisionCalculator, RecallCalculator, F1Calculator
from src.metrics.summarization import (
    Rouge1Calculator,
    Rouge2Calculator,
    RougeLCalculator,
    LLMJudgeCalculator,
)
from src.metrics.text_to_sql import ExecutionAccuracyCalculator, ResultSetAccuracyCalculator


DEFAULT_METRICS: Dict[str, list[Type[MetricCalculator]]] = {
    "classification": [AccuracyCalculator, PrecisionCalculator, RecallCalculator, F1Calculator],
    "text_to_sql": [ExecutionAccuracyCalculator, ResultSetAccuracyCalculator],
}

# Strategy-specific metric mappings for evaluators that support multiple strategies
STRATEGY_METRICS: Dict[str, Dict[str, list[Type[MetricCalculator]]]] = {
    "summarization": {
        "semantic": [Rouge1Calculator, Rouge2Calculator, RougeLCalculator],
        "llm_judge": [LLMJudgeCalculator],
    },
}


def get_default_calculators(evaluator: str, strategy: str | None = None) -> list[MetricCalculator]:
    """Resolve default metric calculators for an evaluator (and optional strategy)."""
    if strategy is not None:
        strategy_map = STRATEGY_METRICS.get(evaluator)
        if strategy_map is not None:
            classes = strategy_map.get(strategy)
            if classes is not None:
                return [cls() for cls in classes]

    try:
        classes = DEFAULT_METRICS[evaluator]
    except KeyError:
        raise ValueError(f"No default metrics registered for evaluator: {evaluator}")
    return [cls() for cls in classes]