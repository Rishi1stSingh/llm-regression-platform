from src.comparator import compare
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
