"""Golden dataset loading and validation utilities.

This module provides functions to load, validate, and manage golden test
datasets for LLM evaluation. The datasets live under ``data/golden/`` and
are committed to the repository so they are always available in CI.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.models import GoldenCase


# Evaluator type -> golden dataset file mapping
GOLDEN_DATASETS: dict[str, str] = {
    "classification": "data/golden/classification_cases.json",
    "summary": "data/golden/summary_cases.json",
    "summarization": "data/golden/summary_cases.json",
    "sql": "data/golden/sql_cases.json",
    "text_to_sql": "data/golden/sql_cases.json",
}

# Required fields for every golden case
REQUIRED_FIELDS = ("id", "input", "expected")

# Optional task-specific fields
OPTIONAL_FIELDS = ("expected_result", "metadata")


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    """Load and validate golden cases from a JSON file.

    Args:
        path: Path to a JSON file containing an array of case objects.

    Returns:
        A list of validated ``GoldenCase`` objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is malformed, not an array, or validation fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {file_path}")

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in golden dataset {file_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(
            f"Golden dataset {file_path} must contain a JSON array, got {type(raw).__name__}"
        )

    return validate_cases(raw, dataset_path=str(file_path))


def validate_cases(
    raw_cases: list[Any], *, dataset_path: str | None = None
) -> list[GoldenCase]:
    """Validate raw case dicts and return GoldenCase objects.

    Checks:
      - Each item is a dict.
      - Each item has non-empty 'id', 'input', 'expected'.
      - All case IDs are unique.

    Raises ValueError on any validation failure.
    """
    if not isinstance(raw_cases, list):
        raise ValueError("Cases must be provided as a list")

    seen_ids: set[str] = set()
    cases: list[GoldenCase] = []

    for index, raw in enumerate(raw_cases):
        label = f"index {index}" if dataset_path is None else f"{dataset_path}[{index}]"

        if not isinstance(raw, dict):
            raise ValueError(
                f"Case at {label} is not a JSON object: {type(raw).__name__}"
            )

        case_id = raw.get("id")
        if not case_id or not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case at {label} is missing a non-empty 'id' field")

        if case_id in seen_ids:
            raise ValueError(
                f"Duplicate case ID '{case_id}' found at {label}. "
                "All case IDs must be unique."
            )
        seen_ids.add(case_id)

        for field_name in REQUIRED_FIELDS:
            value = raw.get(field_name)
            if value is None:
                raise ValueError(
                    f"Case '{case_id}' (at {label}) is missing required field '{field_name}'"
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Case '{case_id}' (at {label}) has empty required field '{field_name}'"
                )

        case_kwargs: dict[str, Any] = {
            "id": case_id,
            "input": raw["input"],
            "expected": raw["expected"],
        }
        if "expected_result" in raw:
            case_kwargs["expected_result"] = raw["expected_result"]
        if "metadata" in raw:
            case_kwargs["metadata"] = raw["metadata"]

        cases.append(GoldenCase(**case_kwargs))

    return cases


def load_dataset_by_type(
    dataset_type: str, root: str | Path | None = None
) -> list[GoldenCase]:
    """Load a golden dataset by evaluator/dataset type name.

    Args:
        dataset_type: One of 'classification', 'summary', 'summarization',
            'sql', or 'text_to_sql'.
        root: Project root path. Defaults to the package's parent directory.

    Raises:
        ValueError: If dataset type is unknown.
        FileNotFoundError: If the dataset file does not exist.
    """
    if dataset_type not in GOLDEN_DATASETS:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Available: {', '.join(sorted(GOLDEN_DATASETS.keys()))}"
        )

    dataset_rel = GOLDEN_DATASETS[dataset_type]
    if root is None:
        root = Path(__file__).resolve().parents[1]
    dataset_path = Path(root) / dataset_rel

    return load_golden_cases(dataset_path)


def ensure_sql_database(
    db_path: str | Path, schema_path: str | Path | None = None
) -> None:
    """Create or overwrite a SQLite database from a SQL schema/seed file.

    Used by the text_to_sql evaluator to set up the evaluation database
    from version-controlled seed data.
    """
    if schema_path is None:
        schema_path = (
            Path(__file__).resolve().parents[1] / "data" / "golden" / "sql_schema.sql"
        )

    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"SQL schema file not found: {schema_path}")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            # Another connection (e.g. a concurrent run in the same process)
            # still holds the file open on Windows. The existing database was
            # built from this same schema file, so reuse it instead of failing.
            return

    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()