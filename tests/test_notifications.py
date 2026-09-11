"""Tests for Slack notification behavior (HTTP fully mocked)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from src.models import CaseResult, EvaluationRun
from src.notifications import (
    SLACK_FAILED,
    SLACK_SENT,
    SLACK_SKIPPED,
    build_ci_context,
    notify_slack,
)

WEBHOOK = "https://hooks.slack.com/services/T000/B000/secret"


def _make_run(**kwargs) -> EvaluationRun:
    cases = kwargs.pop(
        "cases",
        [
            CaseResult("C1", "in", "billing", "billing", True),
            CaseResult("C2", "in", "billing", "technical", False),
        ],
    )
    run = EvaluationRun(
        version=kwargs.pop("version", "v1"),
        evaluator=kwargs.pop("evaluator", "classification"),
        cases=cases,
        **kwargs,
    )
    run.finalize_score()
    return run


@pytest.fixture
def webhook_env(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK)
    return monkeypatch


def _mock_post(monkeypatch, *, exc=None, status_code=200):
    post = MagicMock()
    if exc is not None:
        post.side_effect = exc
    else:
        response = MagicMock()
        response.status_code = status_code
        response.raise_for_status.return_value = None
        post.return_value = response
    monkeypatch.setattr("src.notifications.requests.post", post)
    return post


def test_notify_sends_when_webhook_set(webhook_env, monkeypatch):
    post = _mock_post(monkeypatch)
    status = notify_slack(_make_run(), Path("report.html"))
    assert status == SLACK_SENT
    assert post.call_count == 1
    assert post.call_args.args[0] == WEBHOOK
    text = post.call_args.kwargs["json"]["text"]
    assert "PASS" in text
    assert "total 2 | passed 1 | failed 1 | pass rate 50%" in text


def test_notify_message_includes_ci_context(webhook_env, monkeypatch):
    for key, value in [
        ("GITHUB_REPOSITORY", "Rishi1stSingh/llm-regression-platform"),
        ("GITHUB_REF_NAME", "test-llm-regression"),
        ("GITHUB_SHA", "abc123"),
        ("GITHUB_RUN_ID", "42"),
        ("GITHUB_SERVER_URL", "https://github.com"),
    ]:
        monkeypatch.setenv(key, value)
    post = _mock_post(monkeypatch)
    notify_slack(_make_run(), Path("report.html"), ci_context=build_ci_context())
    text = post.call_args.kwargs["json"]["text"]
    assert "Rishi1stSingh/llm-regression-platform" in text
    assert "test-llm-regression" in text
    assert "abc123" in text
    assert "actions/runs/42" in text


def test_notify_reports_regression(webhook_env, monkeypatch):
    post = _mock_post(monkeypatch)
    run = _make_run(regressions=1)
    notify_slack(run, Path("report.html"))
    assert "REGRESSION DETECTED" in post.call_args.kwargs["json"]["text"]


def test_notify_message_includes_sql_breakdown(webhook_env, monkeypatch):
    post = _mock_post(monkeypatch)
    cases = [
        CaseResult("SQL_001", "q", "s", "s", True),
        CaseResult("SQL_002", "q", "s", "s", True),
        CaseResult("SQL_003", "q", "s", "s", False),
        CaseResult("C1", "in", "billing", "billing", True),
    ]
    notify_slack(_make_run(cases=cases), Path("report.html"))
    text = post.call_args.kwargs["json"]["text"]
    assert "*SQL evaluation:* total 3 | passed 2 | failed 1 | pass rate 67%" in text
    assert "*Summary evaluation:* total 4" in text


def test_notify_message_includes_baseline_version(webhook_env, monkeypatch):
    post = _mock_post(monkeypatch)
    run = _make_run(version="v2", baseline_score=0.5)
    run.delta = run.score - 0.5
    notify_slack(run, Path("report.html"), baseline_version="v1")
    text = post.call_args.kwargs["json"]["text"]
    assert "Version: `v2` (baseline: `v1`)" in text
    assert "Baseline score: 50.0%" in text


def test_notify_skipped_when_no_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    post = _mock_post(monkeypatch)
    status = notify_slack(_make_run(), Path("report.html"))
    assert status == SLACK_SKIPPED
    post.assert_not_called()


def test_notify_http_failure_does_not_raise_or_mutate_run(webhook_env, monkeypatch, capsys):
    error = requests.exceptions.HTTPError("500 for url: " + WEBHOOK)
    error.response = MagicMock(status_code=500)
    post = _mock_post(monkeypatch, exc=error)
    run = _make_run()
    status = notify_slack(run, Path("report.html"))
    assert status == SLACK_FAILED
    assert run.status == "passed"  # evaluation result unchanged
    captured = capsys.readouterr()
    assert "Slack notification failed" in captured.err
    assert WEBHOOK not in captured.err  # webhook secret must never be logged


def test_notify_connection_error_does_not_raise(webhook_env, monkeypatch):
    post = _mock_post(
        monkeypatch, exc=requests.exceptions.ConnectionError("network down")
    )
    status = notify_slack(_make_run(), Path("report.html"))
    assert status == SLACK_FAILED
    post.assert_called_once()


def _make_cli_args(no_slack: bool) -> SimpleNamespace:
    return SimpleNamespace(
        version="v1",
        baseline_version=None,
        provider=None,
        mock=True,
        no_slack=no_slack,
        cases=None,
        concurrency=5,
    )


def test_cli_no_slack_flag_skips_notification(monkeypatch):
    notify = MagicMock()
    monkeypatch.setattr("src.cli.notify_slack", notify)
    from src.cli import run_evaluation

    run_evaluation(_make_cli_args(no_slack=True))
    notify.assert_not_called()


def test_cli_sends_notification_when_not_disabled(webhook_env, monkeypatch, capsys):
    post = _mock_post(monkeypatch)
    from src.cli import run_evaluation

    run_evaluation(_make_cli_args(no_slack=False))
    assert post.call_count == 1
    assert post.call_args.args[0] == WEBHOOK
    assert "Slack notification sent." in capsys.readouterr().out
