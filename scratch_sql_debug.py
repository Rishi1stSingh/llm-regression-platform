import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sql_executor import SQLiteExecutor
from src.golden import ensure_sql_database

GOLDEN = ROOT / "data" / "golden"
SCHEMA = GOLDEN / "sql_schema.sql"

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
DB_PATH = tmp.name
ensure_sql_database(Path(DB_PATH))

exp_sql = "SELECT COUNT(*) FROM customers"
gen_sql = "SELECT COUNT(*) AS total_customers FROM customers;"

exp_exec = SQLiteExecutor(Path(DB_PATH))
gen_exec = SQLiteExecutor(Path(DB_PATH))

print("===== SQL_001 =====")
exp_success, exp_rows, exp_err = (True, None, None)
gen_success, gen_rows, gen_err = (True, None, None)
try:
    exp_rows = exp_exec.execute(exp_sql)
    exp_success, exp_err = True, None
except Exception as e:
    exp_success, exp_err = False, repr(e)
try:
    gen_rows = gen_exec.execute(gen_sql)
    gen_success, gen_err = True, None
except Exception as e:
    gen_success, gen_err = False, repr(e)

print("EXPECTED SQL:", exp_sql)
print("GENERATED SQL:", gen_sql)
print()
print("EXPECTED execution_success:", exp_success)
print("EXPECTED execution_error:", exp_err)
print("EXPECTED actual_rows:", exp_rows)
print()
print("GENERATED execution_success:", gen_success)
print("GENERATED execution_error:", gen_err)
print("GENERATED actual_rows:", gen_rows)
print()
print("EXPECTED rows (repr):", repr(exp_rows))
print("GENERATED rows (repr):", repr(gen_rows))
print()
print("EXPECTED row[0] types:", [type(r[0]) for r in exp_rows] if exp_rows else None)
print("GENERATED row[0] types:", [type(r[0]) for r in gen_rows] if gen_rows else None)
print()
print("EXPECTED set normalized:", set(tuple(r) for r in exp_rows) if exp_rows else None)
print("GENERATED set normalized:", set(tuple(r) for r in gen_rows) if gen_rows else None)
print()
print("Result set equal (value-based, normalized)?", set(tuple(r) for r in exp_rows) == set(tuple(r) for r in gen_rows))
print()
print("=== TextToSQLEvaluator._results_equal(actual, expected) ===")
print("actual:", gen_rows, "  (type:", type(gen_rows), ")")
expected_val = [[8]]
print("expected:", expected_val, "  (type:", type(expected_val), ")")
from src.evaluators.text_to_sql import TextToSQLEvaluator
print("equal?", TextToSQLEvaluator._results_equal(gen_rows, expected_val))
