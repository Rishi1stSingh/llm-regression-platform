"""Regression tests: evaluator types must be handled independently.

Guards against the reported bug where summary/text_to_sql runs silently used
classification baselines and classification metric thresholds, producing
classification-style metrics/outputs in non-classification runs (terminal and
HTML reports).
"""

import pytest

from src.cli import run_evaluation
from src.comparator import compare
from src.models import CaseResult, EvaluationRun
from src.metrics.base import MetricResult
from src.storage import SQLiteRepository
from types import SimpleNamespace


def _make_args(dataset=None, evaluator=None, version="v1", baseline_version=None, mock=True):
    args = SimpleNamespace(
        version=version,
        baseline_version=baseline_version,
        provider=None,
        mock=mock,
        no_slack=True,
        cases=None,
        concurrency=5,
        dataset=dataset,
        evaluator=evaluator,
    )
    setattr(args, "async", False)
    return args


def _make_run(version, evaluator):
    run = EvaluationRun(version, evaluator, [])
    run.metrics = [MetricResult("rouge_1", 0.5)]
    return run


def _patch_db(monkeypatch, tmp_path):
    from src.storage import SQLiteRepository as _Repo

    db_path = tmp_path / "eval.db"
    monkeypatch.setattr("src.cli.SQLiteRepository", lambda path: _Repo(db_path))
    return db_path


# ---------------------------------------------------------------------------
# §2/§9 Evaluator selection must propagate into baseline comparison
# ---------------------------------------------------------------------------


def test_baseline_lookup_filters_by_evaluator(tmp_path):
    repo = SQLiteRepository(tmp_path / "eval.db")
    class_baseline = _make_run("v1", "classification")
    class_baseline.finalize_score()
    repo.save_run(class_baseline)

    # A summarization baseline of the same version does NOT exist — the lookup
    # restricted to 'summarization' must not return the classification run.
    assert repo.latest_successful_run("v1", evaluator="summarization") is None
    assert repo.latest_successful_run("v1", evaluator="classification") is not None
    # Legacy call style (no evaluator) keeps old behavior.
    assert repo.latest_successful_run("v1") is not None


def test_compare_refuses_cross_evaluator_baseline():
    baseline = _make_run("v1", "classification")
    baseline.finalize_score()
    candidate = _make_run("v2", "summarization")
    candidate.finalize_score()

    with pytest.raises(ValueError, match="different evaluator types"):
        compare(candidate, baseline, max_score_drop=0.02, fail_on_new_regressions=True)


def test_compare_allows_same_evaluator_baseline():
    baseline = _make_run("v1", "text_to_sql")
    baseline.finalize_score()
    candidate = _make_run("v2", "text_to_sql")
    candidate.finalize_score()

    result = compare(candidate, baseline, max_score_drop=0.02, fail_on_new_regressions=True)

    assert result.baseline_run_id == baseline.run_id
    assert result.baseline_score == baseline.score


def test_cli_summary_run_never_uses_classification_baseline(capsys, tmp_path, monkeypatch):
    """The reported bug: summary run fails on classification 'accuracy'/'f1'
    thresholds because its baseline was a classification run."""
    db_path = _patch_db(monkeypatch, tmp_path)
    class_baseline = _make_run("v1", "classification")
    class_baseline.finalize_score()
    class_baseline.metrics = [MetricResult("accuracy", 0.8), MetricResult("f1", 0.8)]
    SQLiteRepository(db_path).save_run(class_baseline)

    code = run_evaluation(_make_args(dataset="summary", baseline_version="v1"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: summarization" in out
    # No classification metric FAILs for a summarization run
    assert "accuracy: FAIL" not in out
    assert "f1: FAIL" not in out
    assert "SKIPPED (no metric thresholds for evaluator 'summarization')" in out


def test_cli_sql_run_never_uses_classification_baseline(capsys, tmp_path, monkeypatch):
    db_path = _patch_db(monkeypatch, tmp_path)
    class_baseline = _make_run("v1", "classification")
    class_baseline.finalize_score()
    class_baseline.metrics = [MetricResult("accuracy", 0.8), MetricResult("f1", 0.8)]
    SQLiteRepository(db_path).save_run(class_baseline)

    run_evaluation(_make_args(dataset="sql", baseline_version="v1"))

    out = capsys.readouterr().out
    assert "Evaluator: text_to_sql" in out
    assert "accuracy: FAIL" not in out
    assert "f1: FAIL" not in out


def test_cli_sql_run_compares_against_sql_baseline(capsys, tmp_path, monkeypatch):
    db_path = _patch_db(monkeypatch, tmp_path)
    sql_baseline = _make_run("v1", "text_to_sql")
    sql_baseline.finalize_score()
    sql_baseline.metrics = [MetricResult("execution_accuracy", 1.0)]
    SQLiteRepository(db_path).save_run(sql_baseline)

    code = run_evaluation(_make_args(dataset="sql", baseline_version="v1"))

    out = capsys.readouterr().out
    assert "Evaluator: text_to_sql" in out
    assert "Delta:" in out  # same-type baseline found, delta computed
    # Classification thresholds are skipped, not failed, for a SQL run
    assert "accuracy: FAIL" not in out
    assert "accuracy: SKIPPED (metric not produced by evaluator 'text_to_sql')" in out


# ---------------------------------------------------------------------------
# §3/§8 Metrics and terminal output must be evaluator-specific
# ---------------------------------------------------------------------------


def test_cli_summary_metrics_are_not_classification_metrics(capsys, tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="summary"))

    out = capsys.readouterr().out
    assert "Evaluator: summarization" in out
    assert "ROUGE-1:" in out
    # Classification metrics must NOT be displayed for summarization
    assert "Accuracy:" not in out
    assert "Precision:" not in out
    assert "Recall:" not in out
    assert "F1:" not in out


def test_cli_sql_metrics_are_not_classification_metrics(capsys, tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="sql"))

    out = capsys.readouterr().out
    assert "Evaluator: text_to_sql" in out
    assert "Execution Accuracy:" in out
    # Standalone classification metric lines must not appear
    assert "\nAccuracy:" not in out
    assert "Precision:" not in out
    assert "Recall:" not in out
    assert "ROUGE" not in out


def test_cli_classification_metrics_unchanged(capsys, tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="classification"))

    out = capsys.readouterr().out
    assert "Evaluator: classification" in out
    assert "Accuracy:" in out
    assert "Precision:" in out
    assert "Recall:" in out
    assert "F1:" in out
    assert "ROUGE" not in out


# ---------------------------------------------------------------------------
# §7 HTML report must contain only the current evaluator's data
# ---------------------------------------------------------------------------


def _read_report(out):
    report_line = next(line for line in out.splitlines() if line.startswith("Report:"))
    with open(report_line.replace("Report: ", ""), encoding="utf-8") as handle:
        return handle.read()


def test_summary_html_contains_only_summary_data(capsys, tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="summary"))

    html = _read_report(capsys.readouterr().out)
    assert "Evaluator:</b> summarization" in html
    assert "Rouge_1" in html
    # Classification metric names must not appear as metric rows
    assert "<th style='text-align:left;padding:6px'>Accuracy</th>" not in html
    assert "<th style='text-align:left;padding:6px'>Precision</th>" not in html
    # Summary case IDs, not classification ones
    assert "SUM_001" in html
    assert ">001<" not in html
    assert ">002<" not in html


def test_sql_html_contains_only_sql_data(capsys, tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="sql"))

    html = _read_report(capsys.readouterr().out)
    assert "Evaluator:</b> text_to_sql" in html
    assert "Execution_accuracy" in html
    assert "<th style='text-align:left;padding:6px'>Accuracy</th>" not in html
    assert "<th style='text-align:left;padding:6px'>Precision</th>" not in html
    assert "SQL_001" in html
    assert ">001<" not in html


def test_classification_html_contains_classification_data(capsys, tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="classification"))

    html = _read_report(capsys.readouterr().out)
    assert "Evaluator:</b> classification" in html
    assert "Accuracy" in html
    assert "Precision" in html
    assert "Rouge" not in html
    assert "Execution_accuracy" not in html


# ---------------------------------------------------------------------------
# §11 Dataset mapping
# ---------------------------------------------------------------------------


def test_golden_datasets_mapping_is_complete():
    from src.golden import GOLDEN_DATASETS

    assert GOLDEN_DATASETS["classification"].endswith("classification_cases.json")
    assert GOLDEN_DATASETS["summary"] == GOLDEN_DATASETS["summarization"]
    assert GOLDEN_DATASETS["summary"].endswith("summary_cases.json")
    assert GOLDEN_DATASETS["sql"] == GOLDEN_DATASETS["text_to_sql"]
    assert GOLDEN_DATASETS["sql"].endswith("sql_cases.json")


@pytest.mark.parametrize(
    "dataset_arg, evaluator_name",
    [
        ("classification", "classification"),
        ("summary", "summarization"),
        ("sql", "text_to_sql"),
    ],
)
def test_cli_dataset_mapping_selects_correct_evaluator(
    capsys, tmp_path, monkeypatch, dataset_arg, evaluator_name
):
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset=dataset_arg))

    assert f"Evaluator: {evaluator_name}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# §11 Cross-evaluator isolation: sequential runs must not leak state
# ---------------------------------------------------------------------------


def test_sequential_evaluator_runs_are_isolated(capsys, tmp_path, monkeypatch):
    """classification -> summary -> SQL in the same process: each run must
    produce its own evaluator name, metrics, and case outputs."""
    _patch_db(monkeypatch, tmp_path)

    run_evaluation(_make_args(dataset="classification"))
    class_out = capsys.readouterr().out
    run_evaluation(_make_args(dataset="summary"))
    sum_out = capsys.readouterr().out
    run_evaluation(_make_args(dataset="sql"))
    sql_out = capsys.readouterr().out

    assert "Evaluator: classification" in class_out and "Accuracy:" in class_out
    assert "Evaluator: summarization" in sum_out and "ROUGE-1:" in sum_out
    assert "Accuracy:" not in sum_out
    assert "Evaluator: text_to_sql" in sql_out and "Execution Accuracy:" in sql_out
    assert "\nAccuracy:" not in sql_out and "Precision:" not in sql_out
    # No foreign case IDs in any run's output
    assert "SUM_001" not in class_out
    assert "SQL_001" not in class_out
    assert "SUM_001" not in sql_out


def test_sql_output_is_sql_and_summary_output_is_summary(capsys, tmp_path, monkeypatch):
    """Case inputs must come from the current evaluator's dataset — not shared
    between evaluator runs."""
    from src.evaluators import engine as engine_module

    _patch_db(monkeypatch, tmp_path)
    engine_cases = {}

    original_run = engine_module.EvaluationEngine.run

    def tracking_run(self, cases, version):
        engine_cases["run"] = list(cases)
        return original_run(self, cases, version)

    monkeypatch.setattr(engine_module.EvaluationEngine, "run", tracking_run)

    run_evaluation(_make_args(dataset="summary"))
    summary_run_cases = engine_cases["run"]
    run_evaluation(_make_args(dataset="sql"))
    sql_run_cases = engine_cases["run"]

    assert all(case.id.startswith("SUM_") for case in summary_run_cases)
    assert all(case.id.startswith("SQL_") for case in sql_run_cases)
    # Inputs are not shared between evaluator runs
    assert {case.input for case in summary_run_cases}.isdisjoint(
        {case.input for case in sql_run_cases}
    )


# ---------------------------------------------------------------------------
# §4 System prompts must be task-specific (real-provider bug: the classification
# prompt made summary/SQL models output ticket labels like 'billing')
# ---------------------------------------------------------------------------


def test_prompt_files_are_task_specific(tmp_path):
    """The classification prompt demands a ticket label; the summary/SQL
    prompts must instruct their own task instead."""
    from pathlib import Path

    from src.cli import PROMPT_FILES

    root = Path(__file__).resolve().parents[1] / "prompts"
    for version in ("v1", "v2"):
        class_prompt = (root / PROMPT_FILES["classification"].format(version=version)).read_text()
        sum_prompt = (root / PROMPT_FILES["summarization"].format(version=version)).read_text()
        sql_prompt = (root / PROMPT_FILES["text_to_sql"].format(version=version)).read_text()

        # Classification prompt keeps the label-based instructions
        assert "billing" in class_prompt
        assert "classifier" in class_prompt
        assert "label" in class_prompt
        # Summary/SQL prompts must NOT be the classification prompt
        assert "billing" not in sum_prompt
        assert "Return exactly one label" not in sum_prompt
        assert "billing" not in sql_prompt
        assert "Return exactly one label" not in sql_prompt
        # And each prompt is about its own task
        assert "summar" in sum_prompt.lower()
        assert "sql" in sql_prompt.lower()


def test_summary_run_uses_summarization_system_prompt(capsys, tmp_path, monkeypatch):
    """End-to-end: with a (mocked) real provider, the summary evaluator must
    receive the summarization system prompt, not the classification one."""
    _patch_db(monkeypatch, tmp_path)
    seen = {}

    class RecordingGroq:
        def __init__(self, **kwargs):
            pass

        def summarize(self, system_prompt, document):
            seen["system_prompt"] = system_prompt
            return "The report shows a 15% revenue increase to $12.5 million."

    monkeypatch.setattr("src.cli.GroqLLMClient", RecordingGroq)

    code = run_evaluation(_make_args(dataset="summary", mock=False))

    capsys.readouterr()
    assert code == 0
    assert "summar" in seen["system_prompt"].lower()
    assert "billing" not in seen["system_prompt"].lower()
    assert "Return exactly one label" not in seen["system_prompt"]


def test_sql_run_uses_sql_system_prompt(capsys, tmp_path, monkeypatch):
    """End-to-end: with a (mocked) real provider, the SQL evaluator must
    receive the text-to-SQL system prompt, not the classification one."""
    _patch_db(monkeypatch, tmp_path)
    seen = {}

    class RecordingGroq:
        def __init__(self, **kwargs):
            pass

        def generate_sql(self, system_prompt, question):
            seen["system_prompt"] = system_prompt
            return "SELECT COUNT(*) FROM customers;"

    monkeypatch.setattr("src.cli.GroqLLMClient", RecordingGroq)

    code = run_evaluation(_make_args(dataset="sql", mock=False))

    capsys.readouterr()
    assert code == 0
    assert "sql" in seen["system_prompt"].lower()
    assert "billing" not in seen["system_prompt"].lower()
    assert "Return exactly one label" not in seen["system_prompt"]


def test_results_equal_normalizes_json_lists_and_sqlite_tuples():
    from src.evaluators.text_to_sql import TextToSQLEvaluator

    # JSON-loaded expected results are lists of lists; SQLite yields tuples.
    assert TextToSQLEvaluator._results_equal([(8,)], [[8]])
    assert TextToSQLEvaluator._results_equal(
        [("Carol Davis", "carol@example.com"), ("Eva Martinez", "eva@example.com")],
        [["Eva Martinez", "eva@example.com"], ["Carol Davis", "carol@example.com"]],
    )
    # Order does not matter, but different values do
    assert not TextToSQLEvaluator._results_equal([(8,)], [[9]])
    assert TextToSQLEvaluator._results_equal([], [])

