from types import SimpleNamespace

import pytest
from src.cli import run_evaluation


def _make_args(cases=None, version="v2"):
    args = SimpleNamespace(
        version=version,
        baseline_version=None,
        provider=None,
        mock=True,
        no_slack=True,
        cases=cases,
        concurrency=5,
    )
    setattr(args, "async", False)
    return args


def test_no_cases_runs_all(capsys):
    args = _make_args(cases=None)
    run_evaluation(args)
    # Just verify it runs without error - all cases evaluated
    captured = capsys.readouterr()
    assert "Evaluation Results" in captured.out


def test_cases_10_limits_to_10(capsys, monkeypatch):
    # Track how many cases are passed to the engine
    original_run = None
    cases_passed = []

    from src.evaluators import engine as engine_module

    original_run = engine_module.EvaluationEngine.run

    def tracking_run(self, cases, version):
        cases_passed.extend(list(cases))
        return original_run(self, cases, version)

    monkeypatch.setattr(engine_module.EvaluationEngine, "run", tracking_run)

    args = _make_args(cases=10)
    run_evaluation(args)
    assert len(cases_passed) == 10


def test_cases_larger_than_dataset_runs_all(capsys, monkeypatch):
    from src.evaluators import engine as engine_module

    original_run = engine_module.EvaluationEngine.run
    cases_passed = []

    def tracking_run(self, cases, version):
        cases_passed.extend(list(cases))
        return original_run(self, cases, version)

    monkeypatch.setattr(engine_module.EvaluationEngine, "run", tracking_run)

    args = _make_args(cases=99999)
    run_evaluation(args)
    # Should run all available cases (more than 0 but not 99999)
    assert len(cases_passed) > 0
    assert len(cases_passed) < 99999


def test_cases_0_raises_error():
    args = _make_args(cases=0)
    with pytest.raises(ValueError, match="--cases must be a positive integer"):
        run_evaluation(args)


def test_cases_negative_raises_error():
    args = _make_args(cases=-5)
    with pytest.raises(ValueError, match="--cases must be a positive integer"):
        run_evaluation(args)
