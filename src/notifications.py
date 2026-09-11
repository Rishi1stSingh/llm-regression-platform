"""Slack notifications for evaluation runs.

The webhook URL is read from the ``SLACK_WEBHOOK_URL`` environment variable
(populated in CI from the ``SLACK_WEBHOOK_URL`` secret, or from a local
``.env`` file in development).

Slack is a notification mechanism only: failures here are logged and never
change the evaluation result or the process exit code. The webhook URL is a
secret, so it is never printed — not even in exception messages (requests
exception text can embed the full request URL).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

from src.models import CaseResult, EvaluationRun

# Return-status values used by notify_slack
SLACK_SENT = "sent"        # delivered to Slack
SLACK_SKIPPED = "skipped"  # no webhook configured
SLACK_FAILED = "failed"    # webhook configured but delivery failed

_TIMEOUT_SECONDS = 15


def _pass_rate(passed: int, total: int) -> str:
    return f"{passed / total:.0%}" if total else "n/a"


def _case_breakdown(cases: list[CaseResult]) -> tuple[int, int, int, str]:
    total = len(cases)
    passed = sum(1 for case in cases if case.passed)
    failed = total - passed
    return total, passed, failed, _pass_rate(passed, total)


def _is_regression(run: EvaluationRun) -> bool:
    if run.regressions > 0:
        return True
    return any(not check.passed for check in run.metric_regression_checks)


def _status_label(run: EvaluationRun) -> str:
    if _is_regression(run):
        return "REGRESSION DETECTED"
    return "PASS" if run.status == "passed" else "FAIL"


def _sql_section(run: EvaluationRun) -> str | None:
    sql_cases = [case for case in run.cases if case.case_id.upper().startswith("SQL_")]
    if not sql_cases:
        return None
    total, passed, failed, rate = _case_breakdown(sql_cases)
    return (
        f"*SQL evaluation:* total {total} | passed {passed} | "
        f"failed {failed} | pass rate {rate}"
    )


def build_slack_message(
    run: EvaluationRun,
    report_path: Path,
    ci_context: dict[str, Any] | None = None,
    baseline_version: str | None = None,
) -> str:
    """Build the Slack message body from an EvaluationRun (no evaluation logic)."""
    ci_context = ci_context or {}
    regression = _is_regression(run)
    icon = "🚨" if regression or run.status != "passed" else "✅"

    lines = [f"{icon} *LLM evaluation {_status_label(run)}*"]

    # Current/baseline versions, score and delta
    score = f"Score: {run.score:.1%}"
    if baseline_version:
        lines.append(
            f"Version: `{run.version}` (baseline: `{baseline_version}`) | {score}"
        )
    else:
        lines.append(f"Version: `{run.version}` | {score}")
    if run.baseline_score is not None:
        lines.append(f"Baseline score: {run.baseline_score:.1%}")
    if run.delta is not None:
        lines.append(f"Delta: {run.delta:+.1%}")

    # CI context (only populated when running in GitHub Actions)
    if ci_context.get("repository"):
        lines.append(f"Repository: `{ci_context['repository']}`")
    if ci_context.get("pr_number") and ci_context.get("pr_url"):
        lines.append(f"PR: <{ci_context['pr_url']}|#{ci_context['pr_number']}>")
    if ci_context.get("branch"):
        lines.append(f"Branch: `{ci_context['branch']}`")
    if ci_context.get("commit"):
        lines.append(f"Commit: `{ci_context['commit']}`")
    if ci_context.get("run_url"):
        lines.append(
            f"Action: <{ci_context['run_url']}|run #{ci_context.get('run_id', '?')}>"
        )

    # Overall evaluation summary
    total, passed, failed, rate = _case_breakdown(run.cases)
    lines.append(
        f"*Summary evaluation:* total {total} | passed {passed} | "
        f"failed {failed} | pass rate {rate}"
    )

    # Per-evaluator breakdown for the SQL (text_to_sql) golden dataset
    sql_section = _sql_section(run)
    if sql_section:
        lines.append(sql_section)

    lines.append(f"Regressions: {run.regressions} | Improvements: {run.improvements}")
    lines.append(f"Report generated: `{report_path.name}`")
    return "\n".join(lines)


def build_ci_context() -> dict[str, Any]:
    """Collect GitHub Actions context from environment variables.

    Every field is optional; outside CI the dict is empty and those message
    lines are simply omitted. Never includes secrets.
    """
    repository = os.environ.get("GITHUB_REPOSITORY")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID")
    branch = os.environ.get("GITHUB_REF_NAME")

    context: dict[str, Any] = {
        "repository": repository,
        "branch": branch,
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": run_id,
    }
    if repository and run_id:
        context["run_url"] = f"{server}/{repository}/actions/runs/{run_id}"

    # Pull-request runs expose GITHUB_REF_NAME as "<number>/merge"
    match = re.match(r"^(\d+)/merge$", branch or "")
    if match and repository:
        pr_number = match.group(1)
        context["pr_number"] = pr_number
        context["pr_url"] = f"{server}/{repository}/pull/{pr_number}"
        context["branch"] = None  # a "<number>/merge" ref is not a branch name

    return {key: value for key, value in context.items() if value}


def notify_slack(
    run: EvaluationRun,
    report_path: Path,
    ci_context: dict[str, Any] | None = None,
    baseline_version: str | None = None,
) -> str:
    """Send the evaluation notification to Slack.

    Reads the webhook from the ``SLACK_WEBHOOK_URL`` environment variable.
    Never raises: delivery problems are logged to stderr and reported through
    the return value so they cannot turn a passing evaluation into a failure.

    Returns:
        SLACK_SENT, SLACK_SKIPPED (no webhook configured) or SLACK_FAILED.
    """
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print(
            "Slack notification skipped: SLACK_WEBHOOK_URL is not set.",
            file=sys.stderr,
        )
        return SLACK_SKIPPED

    text = build_slack_message(run, report_path, ci_context, baseline_version)
    try:
        response = requests.post(webhook, json={"text": text}, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # Log without exc's message: its text can contain the webhook URL,
        # which is a secret and must never reach CI logs.
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f" (HTTP {status_code})" if status_code else ""
        print(
            f"Slack notification failed: {type(exc).__name__}{detail} — "
            "evaluation result is unchanged.",
            file=sys.stderr,
        )
        return SLACK_FAILED
    return SLACK_SENT
