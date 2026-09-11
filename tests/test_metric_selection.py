import pytest

from src.metrics.engine import MetricsEngine
from src.metrics.base import MetricResult
from src.metrics.registry import get_default_calculators
from src.metrics.classification import PrecisionCalculator
from src.models import EvaluationRun, CaseResult


def make_run(pairs: list[tuple[str, str]], evaluator: str = "classification") -> EvaluationRun:
    cases = [CaseResult(str(i + 1), "in", exp, act, exp == act) for i, (exp, act) in enumerate(pairs)]
    run = EvaluationRun("v", evaluator, cases)
    run.finalize_score()
    return run


def test_auto_select_classification_metrics():
    pairs = [("billing", "billing"), ("account", "account"), ("technical", "technical"), ("general", "general")]
    run = make_run(pairs, evaluator="classification")
    engine = MetricsEngine()
    results = engine.compute(run)
    names = {r.name for r in results}
    assert {"accuracy", "precision", "recall", "f1"}.issubset(names)


def test_explicit_calculators_still_work():
    pairs = [("billing", "billing"), ("account", "account")]
    run = make_run(pairs, evaluator="classification")
    # use a single explicit calculator (PrecisionCalculator)
    engine = MetricsEngine(calculators=[PrecisionCalculator()])
    results = engine.compute(run)
    assert len(results) == 1
    assert results[0].name == "precision"


def test_unknown_evaluator_raises():
    run = make_run([("billing", "billing")], evaluator="unknown")
    engine = MetricsEngine()
    with pytest.raises(ValueError) as exc:
        engine.compute(run)
    assert "No default metrics registered for evaluator: unknown" in str(exc.value)
