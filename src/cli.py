from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.comparator import check_metric_regressions, compare
from src.evaluators.engine import AsyncEvaluationEngine, EvaluationEngine
from src.evaluators.registry import resolve_evaluator
from src.golden import ensure_sql_database, load_dataset_by_type, load_golden_cases
from src.llm import GroqLLMClient, MockLLMClient, NvidiaLLMClient
from src.models import EvaluationRun
from src.metrics.engine import MetricsEngine
from src.notifications import (
    SLACK_FAILED,
    SLACK_SENT,
    build_ci_context,
    notify_slack,
)
from src.reporting import generate_html_report
from src.sql_executor import SQLiteExecutor
from src.storage import SQLiteRepository

ROOT = Path(__file__).resolve().parents[1]

# Display order for metrics in CLI output
METRIC_DISPLAY_ORDER = [
    "accuracy", "precision", "recall", "f1",
    "rouge_1", "rouge_2", "rouge_l", "bertscore",
    "faithfulness", "relevance", "completeness", "coherence", "conciseness", "overall",
    "execution_accuracy", "result_set_accuracy",
]

# Human-readable labels for metrics
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "rouge_1": "ROUGE-1",
    "rouge_2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "bertscore": "BERTScore",
    "faithfulness": "Faithfulness",
    "relevance": "Relevance",
    "completeness": "Completeness",
    "coherence": "Coherence",
    "conciseness": "Conciseness",
    "overall": "Overall",
    "execution_accuracy": "Execution Accuracy",
    "result_set_accuracy": "Result Set Accuracy",
}

# Mapping from CLI --dataset type names to evaluator type names. Both the
# short aliases ('summary', 'sql') and the full evaluator names
# ('summarization', 'text_to_sql') are accepted.
DATASET_TO_EVALUATOR = {
    "classification": "classification",
    "summary": "summarization",
    "summarization": "summarization",
    "sql": "text_to_sql",
    "text_to_sql": "text_to_sql",
}

# Valid --evaluator values (evaluator names plus task aliases), mapped to the
# evaluator type. Based on the actual evaluator registry (classification,
# summarization, text_to_sql).
EVALUATOR_ALIASES = {
    "classification": "classification",
    "summary": "summarization",
    "summarization": "summarization",
    "sql": "text_to_sql",
    "text_to_sql": "text_to_sql",
}

SUPPORTED_EVALUATORS = "classification, summarization, text_to_sql"

# System prompt files are task-specific. Sending the classification prompt
# (which demands a ticket label: billing/account/technical/general) to a
# summarization or text-to-SQL model would make it output labels instead of
# summaries or SQL. Each version has a prompt file per task.
PROMPT_FILES = {
    "classification": "{version}.txt",
    "summarization": "summarization_{version}.txt",
    "text_to_sql": "text_to_sql_{version}.txt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM regression evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--version", choices=["v1", "v2"], required=True)
    run_parser.add_argument("--baseline-version", help="Use the latest successful run for this version.")
    run_parser.add_argument(
        "--provider",
        choices=["groq", "nvidia", "mock"],
        default=None,
        help="LLM provider override (default: config.json llm.provider).",
    )
    run_parser.add_argument("--mock", action="store_true", help="Use deterministic offline client.")
    run_parser.add_argument("--no-slack", action="store_true")
    run_parser.add_argument("--async", action="store_true", help="Evaluate cases concurrently for faster execution.")
    run_parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent evaluations when using --async (default: 5).")
    run_parser.add_argument("--cases", type=int, default=None, help="Limit evaluation to the first N golden-dataset cases.")
    run_parser.add_argument(
        "--dataset",
        default=None,
        help="Golden dataset: a built-in type ('classification', 'summary', "
        "'summarization', 'sql', 'text_to_sql') OR a path to your own "
        "golden-dataset JSON file, e.g. ./data/my_cases.json. For a custom "
        "path, --evaluator is required. (default: config.json "
        "evaluator.type, i.e. 'classification').",
    )
    run_parser.add_argument(
        "--evaluator",
        default=None,
        help=f"Evaluator to use with a custom --dataset path. Supported values: "
        f"{SUPPORTED_EVALUATORS} (aliases: 'summary'->summarization, "
        f"'sql'->text_to_sql). Required when --dataset is a file path — the "
        f"framework never guesses an evaluator for an arbitrary dataset. "
        f"Ignored when --dataset is a built-in type name. "
        f"Example: --dataset ./my_cases.json --evaluator summarization",
    )
    args = parser.parse_args()
    return run_evaluation(args)


def run_evaluation(args: argparse.Namespace) -> int:
    if args.cases is not None and args.cases <= 0:
        raise ValueError("--cases must be a positive integer")
    config = json.loads((ROOT / "config.json").read_text())
    # Determine the evaluator type and dataset source.
    #
    # Resolution rules (single source of truth for everything downstream):
    #   --dataset <built-in type name>  -> built-in dataset + mapped evaluator
    #   --dataset <file path>           -> custom dataset; --evaluator REQUIRED
    #   (no --dataset)                  -> config.json evaluator.type + its dataset
    # The framework NEVER silently assumes classification for a custom path.
    dataset_arg = getattr(args, "dataset", None)
    evaluator_arg = getattr(args, "evaluator", None)

    # Validate an explicit --evaluator against the real evaluator registry
    # (classification, summarization, text_to_sql + aliases summary/sql).
    evaluator_type: str | None = None
    if evaluator_arg is not None:
        evaluator_type = EVALUATOR_ALIASES.get(evaluator_arg)
        if evaluator_type is None:
            print(
                f"Error: unknown evaluator '{evaluator_arg}'. "
                f"Supported evaluators: {SUPPORTED_EVALUATORS}."
            )
            return 2

    custom_dataset_path: Path | None = None
    if dataset_arg is not None and dataset_arg in DATASET_TO_EVALUATOR:
        # A built-in type name is authoritative: evaluator AND dataset are
        # taken from the mapping (--evaluator is ignored for type names).
        evaluator_type = DATASET_TO_EVALUATOR[dataset_arg]
    elif dataset_arg is not None:
        # --dataset is treated as a path to the user's own golden dataset.
        custom_dataset_path = Path(dataset_arg).expanduser()
        if not custom_dataset_path.is_absolute():
            custom_dataset_path = Path.cwd() / custom_dataset_path
        custom_dataset_path = custom_dataset_path.resolve()
        if not custom_dataset_path.exists():
            print(f"Error: Golden dataset not found: {custom_dataset_path}")
            return 2
        if evaluator_type is None:
            print(
                "Error: custom dataset path requires --evaluator. "
                f"Supported evaluators: {SUPPORTED_EVALUATORS}. "
                "Example: --dataset ./my_cases.json --evaluator summarization"
            )
            return 2
    elif evaluator_type is None:
        evaluator_type = config.get("evaluator", {}).get("type", "classification")

    # Load the golden dataset (validated, unique IDs): a custom path uses the
    # same loader and validation as the built-in datasets.
    if custom_dataset_path is not None:
        try:
            cases = load_golden_cases(custom_dataset_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            return 2
    else:
        cases = load_dataset_by_type(evaluator_type)
    if args.cases is not None:
        cases = cases[: args.cases]
    # Load the task-specific system prompt for this evaluator type (fall back
    # to the legacy "<version>.txt" file for unknown/custom evaluators).
    prompt_file = PROMPT_FILES.get(evaluator_type, f"{args.version}.txt").format(version=args.version)
    prompt = (ROOT / "prompts" / prompt_file).read_text()
    # Determine provider: --provider > --mock > config llm.provider > nvidia
    provider = getattr(args, "provider", None)
    if args.mock:
        provider = "mock"
    if provider is None:
        provider = config.get("llm", {}).get("provider", "nvidia")
    llm_config = config.get("llm", {})
    requests_per_minute = llm_config.get("requests_per_minute", 25)
    max_retries = llm_config.get("max_retries", 3)

    if provider == "mock":
        client = MockLLMClient()
    elif provider == "groq":
        client = GroqLLMClient(
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
        )
    else:
        client = NvidiaLLMClient(
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
        )
    evaluator_config = config.get("evaluator", {})
    strategy = evaluator_config.get("strategy")

    try:
        if evaluator_type == "summarization":
            evaluator = resolve_evaluator(evaluator_type, client, prompt, strategy=strategy or "semantic")
        elif evaluator_type == "text_to_sql":
            db_path = evaluator_config.get("database") or str(ROOT / "artifacts" / "evaluation.db")
            ensure_sql_database(db_path)
            evaluator = resolve_evaluator(evaluator_type, client, prompt, SQLiteExecutor(db_path))
        elif evaluator_type == "custom":
            evaluator = resolve_evaluator(
                evaluator_type,
                client,
                prompt,
                module=evaluator_config.get("module"),
                **{"class": evaluator_config.get("class")},
            )
        else:
            evaluator = resolve_evaluator(evaluator_type, client, prompt)
    except ValueError as exc:
        print(exc)
        return 2

    if getattr(args, "async", False):
        engine = AsyncEvaluationEngine(evaluator, concurrency=args.concurrency)
    else:
        engine = EvaluationEngine(evaluator)
    run = engine.run(cases, args.version)
    # Release the SQL evaluation database connection: SQLiteExecutor keeps the
    # SQLite file open, which would block a later run's ensure_sql_database()
    # rebuild (Windows file locks) and leak handles across runs.
    executor = getattr(evaluator, "executor", None)
    if executor is not None and hasattr(executor, "close"):
        executor.close()

    # Compute evaluator-specific metrics and attach them to the run
    try:
        metrics = MetricsEngine().compute(run)
    except ValueError as exc:
        print(exc)
        metrics = []
    run.metrics = metrics
    if metrics:
        print("\nEvaluation Results")
        print("------------------")
        print(f"Evaluator: {run.evaluator}")
        if run.strategy:
            print(f"Strategy: {run.strategy}")
        metrics_map = {m.name: m.value for m in metrics}
        for key in METRIC_DISPLAY_ORDER:
            if key in metrics_map:
                label = METRIC_LABELS.get(key, key.capitalize())
                print(f"{label}: {metrics_map[key]:.2%}")
        print("")
    repository = SQLiteRepository(ROOT / "artifacts" / "evaluations.db")
    # Baselines must come from the SAME evaluator type: comparing a summary/SQL
    # run against a classification baseline would mix incompatible scores,
    # metrics, and case outputs.
    baseline = (
        repository.latest_successful_run(args.baseline_version, evaluator=run.evaluator)
        if args.baseline_version
        else None
    )
    settings = config["regression"]
    compare(run, baseline, settings["max_score_drop"], settings["fail_on_new_regressions"])

    # Per-metric regression checks against configured thresholds. Configured
    # thresholds are evaluator-aware: a threshold for a metric the current
    # evaluator does not produce (e.g. 'accuracy' for summarization/text_to_sql
    # runs, which have ROUGE/execution metrics) is skipped, never failed.
    thresholds = settings.get("metrics", [])
    run_metric_names = {metric.name for metric in run.metrics}
    applicable_thresholds = [t for t in thresholds if t.get("name") in run_metric_names]
    not_applicable = [t for t in thresholds if t.get("name") not in run_metric_names]
    run.metric_regression_checks = check_metric_regressions(run, baseline, applicable_thresholds)
    print("\nMetric Regression Checks")
    print("------------------------")
    if not thresholds:
        print("No metric thresholds configured.")
    elif baseline is None and not applicable_thresholds:
        print(f"SKIPPED (no metric thresholds for evaluator '{run.evaluator}')")
    elif baseline is None:
        for threshold in applicable_thresholds:
            print(f"{threshold.get('name')}: SKIPPED (no baseline run)")
        for threshold in not_applicable:
            print(
                f"{threshold.get('name')}: SKIPPED "
                f"(metric not produced by evaluator '{run.evaluator}')"
            )
    else:
        for threshold in not_applicable:
            print(
                f"{threshold.get('name')}: SKIPPED "
                f"(metric not produced by evaluator '{run.evaluator}')"
            )
        for check in run.metric_regression_checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"{check.name}: {status} | {check.message}")
        if any(not check.passed for check in run.metric_regression_checks):
            run.status = "failed"
    print("")

    repository.save_run(run)
    report_path = generate_html_report(run, ROOT / "artifacts" / "reports", metrics=metrics)
    if not args.no_slack:
        slack_status = notify_slack(
            run,
            report_path,
            ci_context=build_ci_context(),
            baseline_version=args.baseline_version,
        )
        if slack_status == SLACK_SENT:
            print("Slack notification sent.")
        elif slack_status == SLACK_FAILED:
            # Slack must never gate CI: the failure is logged above and the
            # evaluation result (and exit code) stays unchanged.
            print("Slack notification failed — evaluation result unchanged.")
    print(f"Status: {run.status.upper()} | Score: {run.score:.1%} | Delta: {run.delta if run.delta is not None else 'n/a'}")
    print(f"Regressions: {run.regressions} | Improvements: {run.improvements}")
    if run.elapsed_seconds is not None:
        avg = run.elapsed_seconds / len(run.cases) if run.cases else 0.0
        print(f"Time: {run.elapsed_seconds:.1f}s total | {avg:.2f}s avg per case | {len(run.cases)} cases")
    print(f"Report: {report_path}")
    return 0 if run.status == "passed" else 1
