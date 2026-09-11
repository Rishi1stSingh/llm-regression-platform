from __future__ import annotations

from src.metrics.engine import MetricsEngine
from src.metrics.text_to_sql import ExecutionAccuracyCalculator, ResultSetAccuracyCalculator
from src.models import CaseResult, EvaluationRun


def make_run(passed_flags: list[bool], reasons: list[str] | None = None) -> EvaluationRun:
    if reasons is None:
        reasons = ["Result set match" if p else "Result set mismatch" for p in passed_flags]
    cases = [
        CaseResult(str(i + 1), "q", "SELECT 1;", "SELECT 1;", p, reasons[i])
        for i, p in enumerate(passed_flags)
    ]
    run = EvaluationRun("v", "text_to_sql", cases)
    run.finalize_score()
    return run


def test_execution_accuracy_all_pass():
    run = make_run([True, True, True, True])
    result = ExecutionAccuracyCalculator().compute(run)
    assert result.name == "execution_accuracy"
    assert result.value == 1.0


def test_execution_accuracy_partial():
    run = make_run([True, True, True, False])
    result = ExecutionAccuracyCalculator().compute(run)
    assert result.value == 0.75


def test_execution_accuracy_empty():
    run = make_run([])
    result = ExecutionAccuracyCalculator().compute(run)
    assert result.value == 0.0


def test_result_set_accuracy():
    run = make_run([True, True, False, False])
    result = ResultSetAccuracyCalculator().compute(run)
    assert result.name == "result_set_accuracy"
    assert result.value == 0.5


def test_auto_select_text_to_sql():
    run = make_run([True, False])
    engine = MetricsEngine()
    results = engine.compute(run)
    names = {r.name for r in results}
    assert {"execution_accuracy", "result_set_accuracy"}.issubset(names)