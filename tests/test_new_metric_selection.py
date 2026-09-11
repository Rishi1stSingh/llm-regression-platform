from __future__ import annotations

import pytest

from src.metrics.engine import MetricsEngine
from src.metrics.registry import get_default_calculators
from src.models import CaseResult, EvaluationRun


def make_run(evaluator: str, strategy: str | None = None) -> EvaluationRun:
    cases = [CaseResult("1", "in", "exp", "act", True)]
    run = EvaluationRun("v", evaluator, cases, strategy=strategy)
    run.finalize_score()
    return run


def test_explicit_metric_selection():
    from src.metrics.summarization import Rouge1Calculator

    run = make_run("summarization", "semantic")
    engine = MetricsEngine(calculators=[Rouge1Calculator()])
    results = engine.compute(run)
    assert len(results) == 1
    assert results[0].name == "rouge_1"


def test_unknown_evaluator():
    run = make_run("unknown")
    engine = MetricsEngine()
    with pytest.raises(ValueError, match="No default metrics registered for evaluator: unknown"):
        engine.compute(run)


def test_unknown_metric_strategy_falls_back():
    # summarization without strategy should raise (no default without strategy)
    run = make_run("summarization", None)
    engine = MetricsEngine()
    with pytest.raises(ValueError, match="No default metrics registered"):
        engine.compute(run)


def test_empty_run_handling():
    run = EvaluationRun("v", "summarization", [], strategy="semantic")
    run.finalize_score()
    engine = MetricsEngine()
    results = engine.compute(run)
    assert all(r.value == 0.0 for r in results)


def test_get_default_calculators_classification():
    calcs = get_default_calculators("classification")
    names = {c.name for c in calcs}
    assert {"accuracy", "precision", "recall", "f1"}.issubset(names)


def test_get_default_calculators_summarization_semantic():
    calcs = get_default_calculators("summarization", "semantic")
    names = {c.name for c in calcs}
    assert {"rouge_1", "rouge_2", "rouge_l"}.issubset(names)


def test_get_default_calculators_summarization_llm_judge():
    calcs = get_default_calculators("summarization", "llm_judge")
    assert len(calcs) == 1
    assert calcs[0].name == "llm_judge"


def test_get_default_calculators_text_to_sql():
    calcs = get_default_calculators("text_to_sql")
    names = {c.name for c in calcs}
    assert {"execution_accuracy", "result_set_accuracy"}.issubset(names)