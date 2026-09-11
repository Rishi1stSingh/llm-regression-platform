from __future__ import annotations

from dataclasses import dataclass

from src.models import EvaluationRun


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float | None
    current: float | None
    delta: float | None
    status: str


@dataclass(frozen=True)
class MetricRegressionCheck:
    name: str
    baseline: float | None
    current: float | None
    delta: float | None
    max_drop: float | None
    passed: bool
    message: str


def compare_metrics(candidate: EvaluationRun, baseline: EvaluationRun | None) -> list[MetricDelta]:
    if baseline is None:
        return []

    baseline_map = {metric.name: metric.value for metric in baseline.metrics}
    current_map = {metric.name: metric.value for metric in candidate.metrics}
    results: list[MetricDelta] = []
    for name in sorted(set(baseline_map) | set(current_map)):
        baseline_value = baseline_map.get(name)
        current_value = current_map.get(name)

        if baseline_value is None:
            results.append(MetricDelta(name=name, baseline=None, current=current_value, delta=None, status="missing_from_baseline"))
            continue
        if current_value is None:
            results.append(MetricDelta(name=name, baseline=baseline_value, current=None, delta=None, status="missing_from_current"))
            continue

        delta = round(current_value - baseline_value, 12)
        if delta > 0:
            status = "improved"
        elif delta < 0:
            status = "decreased"
        else:
            status = "unchanged"
        results.append(MetricDelta(name=name, baseline=baseline_value, current=current_value, delta=delta, status=status))
    return results


def check_metric_regressions(candidate: EvaluationRun, baseline: EvaluationRun | None,
                            thresholds: list[dict] | None = None) -> list[MetricRegressionCheck]:
    if baseline is None or not thresholds:
        return []

    baseline_map = {metric.name: metric.value for metric in baseline.metrics}
    current_map = {metric.name: metric.value for metric in candidate.metrics}
    results: list[MetricRegressionCheck] = []
    for config in thresholds:
        name = config.get("name")
        if not name:
            raise ValueError("Metric regression threshold entries require a 'name'.")
        max_drop = float(config.get("max_drop", 0.0))
        baseline_value = baseline_map.get(name)
        current_value = current_map.get(name)
        if baseline_value is None or current_value is None:
            results.append(MetricRegressionCheck(
                name=name,
                baseline=baseline_value,
                current=current_value,
                delta=None,
                max_drop=max_drop,
                passed=False,
                message=f"Configured metric '{name}' is missing from baseline or current run.",
            ))
            continue

        delta = round(current_value - baseline_value, 12)
        drop = round(baseline_value - current_value, 12)
        passed = drop <= max_drop
        if passed:
            message = f"PASS ({delta:+.2f})"
        else:
            message = f"FAIL ({delta:+.2f}, allowed {(-max_drop):+.2f})"
        results.append(MetricRegressionCheck(
            name=name,
            baseline=baseline_value,
            current=current_value,
            delta=delta,
            max_drop=max_drop,
            passed=passed,
            message=message,
        ))
    return results


def compare(candidate: EvaluationRun, baseline: EvaluationRun | None, max_score_drop: float,
            fail_on_new_regressions: bool) -> EvaluationRun:
    if baseline is None:
        return candidate
    # Cross-evaluator comparison is meaningless: scores, metrics, and case
    # outputs belong to different tasks (e.g. a summarization candidate vs a
    # classification baseline). Refuse it instead of silently mixing results —
    # the CLI prevents this at baseline-selection time.
    if baseline.evaluator != candidate.evaluator:
        raise ValueError(
            f"Cannot compare runs of different evaluator types: "
            f"candidate is '{candidate.evaluator}' but baseline is "
            f"'{baseline.evaluator}'. Provide a baseline from the same evaluator."
        )
    candidate.baseline_run_id = baseline.run_id
    candidate.baseline_score = baseline.score
    candidate.delta = candidate.score - baseline.score
    candidate.metric_deltas = compare_metrics(candidate, baseline)
    baseline_cases = {case.case_id: case for case in baseline.cases}
    for case in candidate.cases:
        old = baseline_cases.get(case.case_id)
        if old is None:
            continue
        case.baseline_actual, case.baseline_passed = old.actual, old.passed
        if old.passed and not case.passed:
            case.change = "regression"
            candidate.regressions += 1
        elif not old.passed and case.passed:
            case.change = "improvement"
            candidate.improvements += 1
        elif case.passed:
            case.change = "unchanged_pass"
        else:
            case.change = "unchanged_fail"
    score_regression = candidate.delta < -max_score_drop
    case_regression = fail_on_new_regressions and candidate.regressions > 0
    candidate.status = "failed" if score_regression or case_regression else "passed"
    return candidate
