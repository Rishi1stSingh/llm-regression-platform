from __future__ import annotations

from typing import Any, Callable, Dict

from src.evaluators.base import BaseEvaluator
from src.evaluators.classification import ClassificationEvaluator
from src.evaluators.custom import load_custom_evaluator
from src.evaluators.summarization import SummarizationEvaluator
from src.evaluators.text_to_sql import TextToSQLEvaluator


EVALUATORS: Dict[str, Callable[..., BaseEvaluator]] = {
    "classification": ClassificationEvaluator,
    "summarization": SummarizationEvaluator,
    "text_to_sql": TextToSQLEvaluator,
}


def resolve_evaluator(name: str, *args: Any, **kwargs: Any) -> BaseEvaluator:
    """Resolve an evaluator name to an instantiated BaseEvaluator.

    Supports:
      - Built-in evaluators: classification, summarization, text_to_sql
      - Custom evaluators: name="custom" with module/class kwargs

    Raises ValueError if the evaluator name is unknown or configuration is invalid.
    """
    if name == "custom":
        module = kwargs.pop("module", None)
        class_name = kwargs.pop("class", None)
        return load_custom_evaluator(module, class_name, *args, **kwargs)

    try:
        cls = EVALUATORS[name]
    except KeyError:
        raise ValueError(f"Unknown evaluator: {name}")
    return cls(*args, **kwargs)