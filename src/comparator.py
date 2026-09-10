from src.models import EvaluationRun


def compare(candidate: EvaluationRun, baseline: EvaluationRun | None, max_score_drop: float,
            fail_on_new_regressions: bool) -> EvaluationRun:
    if baseline is None:
        return candidate
    candidate.baseline_run_id = baseline.run_id
    candidate.baseline_score = baseline.score
    candidate.delta = candidate.score - baseline.score
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
