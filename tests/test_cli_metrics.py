from types import SimpleNamespace

from src.cli import run_evaluation


def test_cli_displays_metrics(capsys):
    args = SimpleNamespace(version="v2", baseline_version=None, provider=None, mock=True, no_slack=True, cases=None, concurrency=5)
    setattr(args, "async", False)
    code = run_evaluation(args)
    captured = capsys.readouterr()
    out = captured.out
    assert "Evaluation Results" in out
    assert "Evaluator: classification" in out
    # basic metrics lines
    assert "Accuracy:" in out
    assert "Precision:" in out
    assert "Recall:" in out
    assert "F1:" in out
    assert isinstance(code, int)


def test_cli_mock_command_runs(capsys):
    args = SimpleNamespace(version="v2", baseline_version="v1", provider=None, mock=True, no_slack=True, cases=None, concurrency=5)
    setattr(args, "async", False)
    code = run_evaluation(args)
    captured = capsys.readouterr()
    out = captured.out
    # should still produce report and status lines
    assert "Report:" in out
    assert "Status:" in out
    assert isinstance(code, int)


def test_cli_metric_regression_checks_are_printed(capsys):
    args = SimpleNamespace(version="v2", baseline_version="v1", provider=None, mock=True, no_slack=True, cases=None, concurrency=5)
    setattr(args, "async", False)
    code = run_evaluation(args)
    captured = capsys.readouterr()
    out = captured.out
    assert "Metric Regression Checks" in out
    assert "accuracy" in out.lower()
    assert isinstance(code, int)
