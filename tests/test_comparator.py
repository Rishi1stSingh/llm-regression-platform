from src.comparator import compare, compare_metrics
from src.metrics.base import MetricResult
from src.models import CaseResult, EvaluationRun


def make_run(passed: bool) -> EvaluationRun:
    run = EvaluationRun("v", "classification", [CaseResult("1", "input", "billing", "billing" if passed else "general", passed)])
    run.finalize_score()
    return run


def test_pass_to_fail_is_a_regression_and_fails_run():
    baseline, candidate = make_run(True), make_run(False)
    compare(candidate, baseline, max_score_drop=0.02, fail_on_new_regressions=True)
    assert candidate.regressions == 1
    assert candidate.cases[0].change == "regression"
    assert candidate.status == "failed"


def test_fail_to_pass_is_an_improvement():
    baseline, candidate = make_run(False), make_run(True)
    compare(candidate, baseline, max_score_drop=0.02, fail_on_new_regressions=True)
    assert candidate.improvements == 1
    assert candidate.status == "passed"


def test_metric_delta_is_unchanged_when_values_match():
    baseline = EvaluationRun("v1", "classification", [])
    baseline.metrics = [MetricResult("accuracy", 0.9), MetricResult("f1", 0.8)]
    candidate = EvaluationRun("v2", "classification", [])
    candidate.metrics = [MetricResult("accuracy", 0.9), MetricResult("f1", 0.8)]

    deltas = compare_metrics(candidate, baseline)

    assert [delta.name for delta in deltas] == ["accuracy", "f1"]
    assert deltas[0].delta == 0.0
    assert deltas[0].status == "unchanged"


def test_metric_delta_tracks_improved_and_decreased_metrics():
    baseline = EvaluationRun("v1", "classification", [])
    baseline.metrics = [MetricResult("accuracy", 0.92), MetricResult("precision", 0.95)]
    candidate = EvaluationRun("v2", "classification", [])
    candidate.metrics = [MetricResult("accuracy", 0.97), MetricResult("precision", 0.90)]

    deltas = compare_metrics(candidate, baseline)
    by_name = {delta.name: delta for delta in deltas}

    assert by_name["accuracy"].delta == 0.05
    assert by_name["accuracy"].status == "improved"
    assert by_name["precision"].delta == -0.05
    assert by_name["precision"].status == "decreased"


def test_metric_delta_handles_missing_metrics():
    baseline = EvaluationRun("v1", "classification", [])
    baseline.metrics = [MetricResult("accuracy", 0.9)]
    candidate = EvaluationRun("v2", "classification", [])
    candidate.metrics = []

    deltas = compare_metrics(candidate, baseline)

    assert len(deltas) == 1
    assert deltas[0].name == "accuracy"
    assert deltas[0].baseline == 0.9
    assert deltas[0].current is None
    assert deltas[0].status == "missing_from_current"


def test_metric_delta_handles_no_metrics():
    baseline = EvaluationRun("v1", "classification", [])
    candidate = EvaluationRun("v2", "classification", [])

    assert compare_metrics(candidate, baseline) == []
