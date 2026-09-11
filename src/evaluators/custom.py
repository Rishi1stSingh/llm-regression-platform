from __future__ import annotations

import importlib
from typing import Any

from src.evaluators.base import BaseEvaluator


def load_custom_evaluator(module: str, class_name: str, *args: Any, **kwargs: Any) -> BaseEvaluator:
    """Dynamically load a custom evaluator class from a module.

    Raises:
        ValueError: if the module is not found, the class is not found,
            or the class does not inherit from BaseEvaluator.
    """
    if not module:
        raise ValueError("Custom evaluator requires a 'module'.")
    if not class_name:
        raise ValueError("Custom evaluator requires a 'class'.")

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ValueError(f"Custom evaluator module not found: {module}") from exc

    try:
        cls = getattr(mod, class_name)
    except AttributeError as exc:
        raise ValueError(f"Custom evaluator class not found: {class_name} in module {module}") from exc

    if not isinstance(cls, type) or not issubclass(cls, BaseEvaluator):
        raise ValueError(
            f"Custom evaluator class '{class_name}' must inherit from BaseEvaluator."
        )

    return cls(*args, **kwargs)