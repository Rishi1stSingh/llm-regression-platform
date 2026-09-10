import os
from pathlib import Path

import requests

from src.models import EvaluationRun


def notify_slack(run: EvaluationRun, report_path: Path) -> bool:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        return False
    icon = "✅" if run.status == "passed" else "🚨"
    delta = "—" if run.delta is None else f"{run.delta:+.1%}"
    text = (f"{icon} *LLM evaluation {run.status.upper()}*\n"
            f"Version: `{run.version}` | Score: {run.score:.1%} | Delta: {delta}\n"
            f"Regressions: {run.regressions} | Improvements: {run.improvements}\n"
            f"Report generated: `{report_path.name}`")
    response = requests.post(webhook, json={"text": text}, timeout=15)
    response.raise_for_status()
    return True
