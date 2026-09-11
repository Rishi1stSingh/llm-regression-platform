from src.metrics.classification import AccuracyCalculator
from src.metrics.engine import MetricsEngine
from src.models import EvaluationRun, CaseResult


def make_run_with_pass_counts(total: int, passed: int) -> EvaluationRun:
    cases = []
    for i in range(total):
        pid = str(i + 1)
        is_pass = i < passed
        cases.append(CaseResult(pid, "input", "exp", "act" if is_pass else "wrong", is_pass))
    run = EvaluationRun("v", "classification", cases)
    run.finalize_score()
    return run


def test_accuracy_all_pass():
    run = make_run_with_pass_counts(4, 4)
    calc = AccuracyCalculator()
    result = calc.compute(run)
    assert result.name == "accuracy"
    assert result.value == 1.0


def test_accuracy_partial_pass():
    run = make_run_with_pass_counts(4, 3)
    calc = AccuracyCalculator()
    result = calc.compute(run)
    assert result.value == 0.75


def test_accuracy_all_fail():
    run = make_run_with_pass_counts(4, 0)
    calc = AccuracyCalculator()
    result = calc.compute(run)
    assert result.value == 0.0


def test_accuracy_empty_run():
    run = make_run_with_pass_counts(0, 0)
    calc = AccuracyCalculator()
    result = calc.compute(run)
    assert result.value == 0.0


def test_accuracy_with_metrics_engine_integration():
    run = make_run_with_pass_counts(4, 3)
    engine = MetricsEngine([AccuracyCalculator()])
    results = engine.compute(run)
    assert len(results) == 1
    assert results[0].name == "accuracy"
    assert results[0].value == 0.75
