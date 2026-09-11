"""Tests for custom golden-dataset support (--dataset PATH).

A user should be able to evaluate their own golden dataset through the CLI
without importing framework code or editing GOLDEN_DATASETS. Custom datasets
must go through the exact same load_golden_cases() validation as the built-in
datasets and flow through the same evaluation/report/database pipeline. A
custom dataset path requires an explicit --evaluator: the framework must never
silently assume classification for an arbitrary dataset.
"""

import json
from types import SimpleNamespace

import pytest

import src.cli as cli_module
from src.cli import EVALUATOR_ALIASES, run_evaluation
from src.evaluators import engine as engine_module

VALID_PAYLOAD = [
    {"id": "CASE_001", "input": "my card was charged twice", "expected": "billing"},
    {"id": "CASE_002", "input": "app crashes on upload", "expected": "technical"},
]


def _make_args(dataset=None, evaluator=None, cases=None, version="v2"):
    args = SimpleNamespace(
        version=version,
        baseline_version=None,
        provider=None,
        mock=True,
        no_slack=True,
        cases=cases,
        concurrency=5,
        dataset=dataset,
        evaluator=evaluator,
    )
    setattr(args, "async", False)
    return args


def _write_dataset(tmp_path, payload, name="my_cases.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _track_engine_cases(monkeypatch):
    """Patch EvaluationEngine.run to record the case IDs it receives."""
    original_run = engine_module.EvaluationEngine.run
    cases_passed = []

    def tracking_run(self, run_cases, version):
        cases_passed.extend(case.id for case in run_cases)
        return original_run(self, run_cases, version)

    monkeypatch.setattr(engine_module.EvaluationEngine, "run", tracking_run)
    return cases_passed


def test_custom_dataset_loads_through_cli(capsys, tmp_path, monkeypatch):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)
    cases_passed = _track_engine_cases(monkeypatch)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 0
    assert cases_passed == ["CASE_001", "CASE_002"]
    assert "Evaluation Results" in out
    assert "Evaluator: classification" in out
    assert "Report:" in out


def test_valid_custom_dataset_passes(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert isinstance(code, int)
    assert "Status:" in out
    assert "Report:" in out


def test_custom_dataset_relative_path(capsys, tmp_path, monkeypatch):
    _write_dataset(tmp_path, VALID_PAYLOAD)
    monkeypatch.chdir(tmp_path)

    code = run_evaluation(_make_args(dataset="./my_cases.json", evaluator="classification"))

    assert code == 0


def test_custom_dataset_absolute_windows_path(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)
    # Path.as_posix() renders the absolute path with forward slashes, which
    # must work the same as the native Windows separators.
    code = run_evaluation(_make_args(dataset=path.as_posix(), evaluator="classification"))

    assert code == 0


def test_custom_dataset_with_cases_limit(capsys, tmp_path, monkeypatch):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)
    cases_passed = _track_engine_cases(monkeypatch)

    run_evaluation(_make_args(dataset=str(path), evaluator="classification", cases=1))

    assert cases_passed == ["CASE_001"]


def test_missing_custom_dataset_file_returns_error(capsys, tmp_path):
    code = run_evaluation(_make_args(dataset=str(tmp_path / "does_not_exist.json")))

    out = capsys.readouterr().out
    assert code == 2
    assert "Error:" in out
    assert "Golden dataset not found" in out


def test_invalid_json_custom_dataset_returns_error(capsys, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 2
    assert "Invalid JSON" in out


def test_duplicate_case_ids_return_error(capsys, tmp_path):
    payload = [
        {"id": "CASE_001", "input": "first", "expected": "billing"},
        {"id": "CASE_001", "input": "second", "expected": "technical"},
    ]
    path = _write_dataset(tmp_path, payload)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 2
    assert "Duplicate case ID" in out


def test_missing_required_field_returns_error(capsys, tmp_path):
    payload = [{"id": "CASE_001", "input": "only input"}]  # no 'expected'
    path = _write_dataset(tmp_path, payload)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 2
    assert "required field 'expected'" in out


def test_empty_required_field_returns_error(capsys, tmp_path):
    payload = [{"id": "CASE_001", "input": "x", "expected": "   "}]
    path = _write_dataset(tmp_path, payload)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 2
    assert "empty required field 'expected'" in out


def test_custom_dataset_overrides_mapped_dataset(capsys, tmp_path, monkeypatch):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)
    cases_passed = _track_engine_cases(monkeypatch)

    def unexpected_load(dataset_type, root=None):
        raise AssertionError(
            f"built-in loader must not be used for a custom path, got {dataset_type!r}"
        )

    monkeypatch.setattr(cli_module, "load_dataset_by_type", unexpected_load)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    assert code == 0
    assert cases_passed == ["CASE_001", "CASE_002"]


def test_dataset_type_name_still_uses_built_in_loader(capsys, tmp_path, monkeypatch):
    called_with = []
    real_load = cli_module.load_dataset_by_type

    def recording_load(dataset_type, root=None):
        called_with.append(dataset_type)
        return real_load(dataset_type, root)

    monkeypatch.setattr(cli_module, "load_dataset_by_type", recording_load)

    code = run_evaluation(_make_args(dataset="summary"))

    out = capsys.readouterr().out
    assert code == 0
    assert called_with == ["summarization"]
    assert "Evaluator: summarization" in out


def test_evaluator_flag_selects_evaluator_for_custom_dataset(capsys, tmp_path):
    path = _write_dataset(
        tmp_path,
        [
            {"id": "SUM_001", "input": "Doc one. Second sentence.", "expected": "Doc one."},
            {"id": "SUM_002", "input": "Doc two. More text.", "expected": "Doc two."},
        ],
        name="summary_cases.json",
    )

    code = run_evaluation(_make_args(dataset=str(path), evaluator="summary"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: summarization" in out
    assert "Evaluation Results" in out


def test_evaluator_flag_ignored_for_type_name_dataset(capsys):
    # --evaluator must not override the evaluator implied by a type-name
    # --dataset: sql selects the text_to_sql evaluator and its dataset.
    code = run_evaluation(_make_args(dataset="sql", evaluator="summary"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: text_to_sql" in out


def test_dataset_less_behavior_unchanged(capsys):
    # args without dataset/evaluator attributes (older callers) must keep
    # working: getattr defaults to None and config selects the evaluator.
    args = SimpleNamespace(
        version="v2",
        baseline_version=None,
        provider=None,
        mock=True,
        no_slack=True,
        cases=None,
        concurrency=5,
    )
    setattr(args, "async", False)

    code = run_evaluation(args)

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: classification" in out


def test_custom_dataset_flows_through_db_and_report(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 0
    # The existing pipeline artifacts: DB save + HTML report + regression gate
    assert "Report:" in out
    assert "Status:" in out
    assert "Regressions:" in out
    assert "Metric Regression Checks" in out


def test_custom_dataset_with_baseline_version(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification", version="v1"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Metric Regression Checks" in out
    assert "SKIPPED (no baseline run)" in out


@pytest.mark.parametrize(
    "payload",
    [
        # not a JSON array
        {"id": "CASE_001", "input": "x", "expected": "y"},
        # item is not an object
        ["CASE_001", "CASE_002"],
        # missing id
        [{"input": "x", "expected": "y"}],
        # non-string expected
        [{"id": "CASE_001", "input": "x", "expected": 42}],
    ],
)
def test_invalid_custom_datasets_are_rejected(capsys, tmp_path, payload):
    path = _write_dataset(tmp_path, payload)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 2
    assert "Error:" in out


# ---------------------------------------------------------------------------
# §11 Custom datasets require an explicit evaluator — never silent classification
# ---------------------------------------------------------------------------

SUMMARY_PAYLOAD = [
    {"id": "SUM_001", "input": "The quick brown fox jumps. Second sentence.", "expected": "The quick brown fox jumps."},
    {"id": "SUM_002", "input": "Another document. With more text.", "expected": "Another document."},
]

SQL_PAYLOAD = [
    {
        "id": "SQL_001",
        "input": "How many customers are there in total?",
        "expected": "SELECT COUNT(*) FROM customers",
        "expected_result": [[8]],
    },
    {
        "id": "SQL_002",
        "input": "What are the names and emails of customers in California?",
        "expected": "SELECT name, email FROM customers WHERE state = 'CA'",
        "expected_result": [["Carol Davis", "carol@example.com"]],
    },
]


def test_builtin_summary_selects_summarization(capsys):
    code = run_evaluation(_make_args(dataset="summary"))

    assert code == 0
    assert "Evaluator: summarization" in capsys.readouterr().out


def test_builtin_sql_selects_text_to_sql(capsys):
    code = run_evaluation(_make_args(dataset="sql"))

    assert code == 0
    assert "Evaluator: text_to_sql" in capsys.readouterr().out


def test_builtin_classification_selects_classification(capsys):
    code = run_evaluation(_make_args(dataset="classification"))

    assert code == 0
    assert "Evaluator: classification" in capsys.readouterr().out


def test_builtin_full_evaluator_type_names_work(capsys):
    # --dataset summarization / text_to_sql must behave like the short aliases
    assert run_evaluation(_make_args(dataset="summarization")) == 0
    assert "Evaluator: summarization" in capsys.readouterr().out
    assert run_evaluation(_make_args(dataset="text_to_sql")) == 0
    assert "Evaluator: text_to_sql" in capsys.readouterr().out


def test_custom_summary_dataset_with_evaluator(capsys, tmp_path):
    path = _write_dataset(tmp_path, SUMMARY_PAYLOAD, name="custom_summary.json")

    code = run_evaluation(_make_args(dataset=str(path), evaluator="summarization"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: summarization" in out
    assert "ROUGE-1:" in out
    # No classification metrics for a custom summarization dataset
    assert "Accuracy:" not in out
    assert "Precision:" not in out


def test_custom_sql_dataset_with_evaluator(capsys, tmp_path):
    path = _write_dataset(tmp_path, SQL_PAYLOAD, name="custom_sql.json")

    code = run_evaluation(_make_args(dataset=str(path), evaluator="text_to_sql"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: text_to_sql" in out
    assert "Execution Accuracy:" in out
    # No classification metrics for a custom SQL dataset
    assert "Precision:" not in out
    assert "\nAccuracy:" not in out


def test_custom_classification_dataset_with_evaluator(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD, name="custom_classification.json")

    code = run_evaluation(_make_args(dataset=str(path), evaluator="classification"))

    out = capsys.readouterr().out
    assert code == 0
    assert "Evaluator: classification" in out
    assert "Accuracy:" in out


def test_custom_dataset_without_evaluator_fails_clearly(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)

    code = run_evaluation(_make_args(dataset=str(path)))

    out = capsys.readouterr().out
    assert code == 2
    assert "requires --evaluator" in out
    assert "classification, summarization, text_to_sql" in out
    # MUST NOT silently run the classification evaluator
    assert "Evaluator: classification" not in out
    assert "Evaluation Results" not in out


def test_invalid_evaluator_fails_clearly(capsys, tmp_path):
    path = _write_dataset(tmp_path, VALID_PAYLOAD)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="something_invalid"))

    out = capsys.readouterr().out
    assert code == 2
    assert "unknown evaluator 'something_invalid'" in out
    assert "classification, summarization, text_to_sql" in out


def test_evaluator_aliases_preserved():
    assert EVALUATOR_ALIASES["summary"] == "summarization"
    assert EVALUATOR_ALIASES["sql"] == "text_to_sql"
    assert EVALUATOR_ALIASES["summarization"] == "summarization"
    assert EVALUATOR_ALIASES["text_to_sql"] == "text_to_sql"
    assert EVALUATOR_ALIASES["classification"] == "classification"


def _capture_engine_evaluator(monkeypatch):
    """Patch EvaluationEngine.run to capture the evaluator instance + its prompt."""
    from src.evaluators import engine as engine_module

    captured = {}

    original_run = engine_module.EvaluationEngine.run

    def tracking_run(self, run_cases, version):
        captured["evaluator"] = getattr(self.evaluator, "name", None)
        captured["prompt"] = getattr(self.evaluator, "prompt", None)
        captured["strategy"] = getattr(self.evaluator, "strategy", None)
        return original_run(self, run_cases, version)

    monkeypatch.setattr(engine_module.EvaluationEngine, "run", tracking_run)
    return captured


def test_custom_summary_dataset_selects_summarization_evaluator_and_prompt(capsys, tmp_path, monkeypatch):
    path = _write_dataset(tmp_path, SUMMARY_PAYLOAD, name="custom_summary.json")
    captured = _capture_engine_evaluator(monkeypatch)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="summarization", version="v1"))

    capsys.readouterr()
    assert code == 0
    assert captured["evaluator"] == "summarization"
    # The custom summarization dataset must receive the SUMMARIZATION prompt,
    # never the classification prompt.
    assert captured["prompt"] == "summarization_v1.txt" or "summarization assistant" in captured["prompt"].lower()
    assert "billing" not in captured["prompt"]


def test_custom_sql_dataset_selects_sql_evaluator_and_prompt(capsys, tmp_path, monkeypatch):
    path = _write_dataset(tmp_path, SQL_PAYLOAD, name="custom_sql.json")
    captured = _capture_engine_evaluator(monkeypatch)

    code = run_evaluation(_make_args(dataset=str(path), evaluator="text_to_sql", version="v1"))

    capsys.readouterr()
    assert code == 0
    assert captured["evaluator"] == "text_to_sql"
    assert "text_to_sql_v1.txt" in captured["prompt"].lower() or "sql" in captured["prompt"].lower()
    assert "billing" not in captured["prompt"]
