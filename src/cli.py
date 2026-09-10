from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.comparator import compare
from src.evaluators import ClassificationEvaluator
from src.llm import MockLLMClient, NvidiaLLMClient
from src.models import EvaluationRun, GoldenCase
from src.notifications import notify_slack
from src.reporting import generate_html_report
from src.storage import SQLiteRepository

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM regression evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--version", choices=["v1", "v2"], required=True)
    run_parser.add_argument("--baseline-version", help="Use the latest successful run for this version.")
    run_parser.add_argument("--mock", action="store_true", help="Use deterministic offline client.")
    run_parser.add_argument("--no-slack", action="store_true")
    args = parser.parse_args()
    return run_evaluation(args)


def run_evaluation(args: argparse.Namespace) -> int:
    config = json.loads((ROOT / "config.json").read_text())
    cases = [GoldenCase(**item) for item in json.loads((ROOT / "data/golden_dataset.json").read_text())]
    prompt = (ROOT / "prompts" / f"{args.version}.txt").read_text()
    client = MockLLMClient() if args.mock else NvidiaLLMClient()
    evaluator = ClassificationEvaluator(client, prompt)
    run = EvaluationRun(version=args.version, evaluator=evaluator.name,
                        cases=[evaluator.evaluate_case(case) for case in cases])
    run.finalize_score()
    repository = SQLiteRepository(ROOT / "artifacts" / "evaluations.db")
    baseline = repository.latest_successful_run(args.baseline_version) if args.baseline_version else None
    settings = config["regression"]
    compare(run, baseline, settings["max_score_drop"], settings["fail_on_new_regressions"])
    repository.save_run(run)
    report_path = generate_html_report(run, ROOT / "artifacts" / "reports")
    if not args.no_slack:
        sent = notify_slack(run, report_path)
        if sent:
            print("Slack notification sent.")
    print(f"Status: {run.status.upper()} | Score: {run.score:.1%} | Delta: {run.delta if run.delta is not None else 'n/a'}")
    print(f"Regressions: {run.regressions} | Improvements: {run.improvements}")
    print(f"Report: {report_path}")
    return 0 if run.status == "passed" else 1
