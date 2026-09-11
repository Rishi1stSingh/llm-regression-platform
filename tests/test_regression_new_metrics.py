from __future__ import annotations

from src.comparator import check_metric_regressions, compare_metrics
from src.metrics.base import MetricResult
from src.models import EvaluationRun


def make_run(metrics: dict[str, float], evaluator: str = "summarization") -> EvaluationRun:
    run = EvaluationRun("v", evaluator, [])
    run.metrics = [MetricResult(name, value) for name, value in metrics.items()]
    return run


def test_metric_improvement():
    baseline = make_run({"rouge_l": 0.80})
    candidate = make_run({"rouge_l": 0.85})
    checks = check_metric_regressions(candidate, baseline, [{"name": "rouge_l", "max_drop": 0.02}])
    assert len(checks) == 1
    assert checks[0].passed is True


def test_metric_unchanged():
    baseline = make_run({"rouge_l": 0.80})
    candidate = make_run({"rouge_l": 0.80})
    checks = check_metric_regressions(candidate, baseline, [{"name": "rouge_l", "max_drop": 0.02}])
    assert checks[0].passed is True
    assert checks[0].delta == 0.0


def test_metric_decrease_within_max_drop():
    baseline = make_run({"rouge_l": 0.80})
    candidate = make_run({"rouge_l": 0.79})
    checks = check_metric_regressions(candidate, baseline, [{"name": "rouge_l", "max_drop": 0.02}])
    assert checks[0].passed is True


def test_metric_decrease_exactly_equal_to_max_drop():
    baseline = make_run({"f1": 0.90})
    candidate = make_run({"f1": 0.88})
    checks = check_metric_regressions(candidate, baseline, [{"name": "f1", "max_drop": 0.02}])
    assert checks[0].passed is True


def test_metric_decrease_beyond_max_drop():
    baseline = make_run({"f1": 0.90})
    candidate = make_run({"f1": 0.87})
    checks = check_metric_regressions(candidate, baseline, [{"name": "f1", "max_drop": 0.02}])
    assert checks[0].passed is False


def test_missing_baseline_metric():
    baseline = make_run({})
    candidate = make_run({"rouge_l": 0.80})
    checks = check_metric_regressions(candidate, baseline, [{"name": "rouge_l", "max_drop": 0.02}])
    assert checks[0].passed is False
    assert checks[0].delta is None


def test_missing_current_metric():
    baseline = make_run({"rouge_l": 0.80})
    candidate = make_run({})
    checks = check_metric_regressions(candidate, baseline, [{"name": "rouge_l", "max_drop": 0.02}])
    assert checks[0].passed is False
    assert checks[0].delta is None


def test_multiple_configured_thresholds():
    baseline = make_run({"rouge_1": 0.90, "rouge_l": 0.85, "faithfulness": 0.95})
    candidate = make_run({"rouge_1": 0.89, "rouge_l": 0.80, "faithfulness": 0.96})
    checks = check_metric_regressions(
        candidate,
        baseline,
        [
            {"name": "rouge_1", "max_drop": 0.02},
            {"name": "rouge_l", "max_drop": 0.02},
            {"name": "faithfulness", "max_drop": 0.02},
        ],
    )
    by_name = {c.name: c for c in checks}
    assert by_name["rouge_1"].passed is True
    assert by_name["rouge_l"].passed is False
    assert by_name["faithfulness"].passed is True


def test_compare_metrics_generic():
    baseline = make_run({"execution_accuracy": 0.90})
    candidate = make_run({"execution_accuracy": 0.85})
    deltas = compare_metrics(candidate, baseline)
    assert len(deltas) == 1
    assert deltas[0].name == "execution_accuracy"
    assert deltas[0].status == "decreased"