from __future__ import annotations

import pytest

from src.evaluators.base import BaseEvaluator
from src.evaluators.registry import resolve_evaluator
from src.llm.client import MockLLMClient
from src.models import CaseResult, GoldenCase
from src.sql_executor import FakeSQLExecutor


def test_summarization_resolves():
    evaluator = resolve_evaluator("summarization", MockLLMClient(), "prompt", strategy="semantic")
    assert evaluator.name == "summarization"
    assert evaluator.strategy == "semantic"


def test_summarization_llm_judge_resolves():
    evaluator = resolve_evaluator("summarization", MockLLMClient(), "prompt", strategy="llm_judge")
    assert evaluator.name == "summarization"
    assert evaluator.strategy == "llm_judge"


def test_text_to_sql_resolves():
    executor = FakeSQLExecutor(result=[(1,)])
    evaluator = resolve_evaluator("text_to_sql", MockLLMClient(), "prompt", executor)
    assert evaluator.name == "text_to_sql"


def test_custom_evaluator_resolves():
    # Use an existing test module with a BaseEvaluator subclass
    evaluator = resolve_evaluator(
        "custom",
        module="tests.test_custom_evaluator",
        **{"class": "FakeCustomEvaluator"},
        multiplier=2,
    )
    assert isinstance(evaluator, BaseEvaluator)


def test_custom_missing_module():
    with pytest.raises(ValueError, match="module not found"):
        resolve_evaluator("custom", module="no.such.module", **{"class": "Foo"})


def test_custom_missing_class():
    with pytest.raises(ValueError, match="class not found"):
        resolve_evaluator("custom", module="tests.test_custom_evaluator", **{"class": "NoSuchClass"})


def test_custom_not_base_evaluator():
    with pytest.raises(ValueError, match="must inherit from BaseEvaluator"):
        resolve_evaluator("custom", module="tests.test_custom_evaluator", **{"class": "NotAnEvaluator"})


def test_custom_missing_module_arg():
    with pytest.raises(ValueError, match="requires a 'module'"):
        resolve_evaluator("custom", **{"class": "Foo"})


def test_custom_missing_class_arg():
    with pytest.raises(ValueError, match="requires a 'class'"):
        resolve_evaluator("custom", module="tests.test_custom_evaluator")


def test_invalid_evaluator_produces_clear_error():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        resolve_evaluator("no-such-evaluator")