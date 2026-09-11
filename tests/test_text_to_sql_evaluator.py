from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.evaluators.text_to_sql import TextToSQLEvaluator
from src.models import GoldenCase
from src.sql_executor import FakeSQLExecutor, SQLiteExecutor


def _make_sqlite_db(tmp_path):
    db_path = tmp_path / "golden.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO customers (id, name) VALUES (1, 'A')")
    conn.execute("INSERT INTO customers (id, name) VALUES (2, 'B')")
    conn.execute("INSERT INTO customers (id, name) VALUES (3, 'C')")
    conn.execute("INSERT INTO customers (id, name) VALUES (4, 'D')")
    conn.execute("INSERT INTO customers (id, name) VALUES (5, 'E')")
    conn.execute("INSERT INTO customers (id, name) VALUES (6, 'F')")
    conn.execute("INSERT INTO customers (id, name) VALUES (7, 'G')")
    conn.execute("INSERT INTO customers (id, name) VALUES (8, 'H')")
    conn.commit()
    conn.close()
    return db_path


def _run_sql_on_executor_in_thread(executor, sql):
    errors: list[Exception] = []
    result: list = []

    def target():
        try:
            result.append(executor.execute(sql))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


class FakeSQLClient:

    def __init__(self, sql: str | None = None) -> None:
        self.sql = sql
        self.calls = 0

    def generate_sql(self, system_prompt: str, question: str) -> str:
        self.calls += 1
        if self.sql is not None:
            return self.sql
        return "SELECT COUNT(*) FROM users;"


def test_correct_generated_sql():
    client = FakeSQLClient(sql="SELECT COUNT(*) FROM users;")
    executor = FakeSQLExecutor(result=[(5,)])
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="How many users?", expected="SELECT COUNT(*) FROM users;", expected_result=[(5,)])
    result = evaluator.evaluate_case(case)
    assert result.passed is True
    assert "Result set match" in result.reason


def test_incorrect_result():
    client = FakeSQLClient(sql="SELECT COUNT(*) FROM users;")
    executor = FakeSQLExecutor(result=[(3,)])
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="How many users?", expected="SELECT COUNT(*) FROM users;", expected_result=[(5,)])
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "Result set mismatch" in result.reason


def test_syntactically_invalid_sql():
    client = FakeSQLClient(sql="SELECT FROM WHERE;")
    executor = FakeSQLExecutor(error=ValueError("SQL execution failed: near \"FROM\": syntax error"))
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="How many users?", expected="SELECT COUNT(*) FROM users;", expected_result=[(5,)])
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "SQL execution failed" in result.reason


def test_unsafe_destructive_sql_rejected():
    client = FakeSQLClient(sql="DROP TABLE users;")
    executor = FakeSQLExecutor(error=ValueError("Destructive SQL statement is not allowed: DROP TABLE users;"))
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="How many users?", expected="SELECT COUNT(*) FROM users;", expected_result=[(5,)])
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "Destructive SQL" in result.reason or "SQL execution failed" in result.reason


def test_multiple_sql_forms_same_result():
    client = FakeSQLClient(sql="SELECT COUNT(id) FROM users;")
    executor = FakeSQLExecutor(result=[(5,)])
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="How many users?", expected="SELECT COUNT(*) FROM users;", expected_result=[(5,)])
    result = evaluator.evaluate_case(case)
    assert result.passed is True
    assert "Result set match" in result.reason


def test_executor_failure():
    client = FakeSQLClient(sql="SELECT * FROM users;")
    executor = FakeSQLExecutor(error=RuntimeError("connection lost"))
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="List users", expected="SELECT * FROM users;", expected_result=[(1, "Alice")])
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "SQL execution failed" in result.reason


def test_mocked_sql_executor_used():
    client = FakeSQLClient(sql="SELECT COUNT(*) FROM users;")
    executor = FakeSQLExecutor(result=[(5,)])
    evaluator = TextToSQLEvaluator(client, "prompt", executor)
    case = GoldenCase(id="1", input="How many users?", expected="SELECT COUNT(*) FROM users;", expected_result=[(5,)])
    evaluator.evaluate_case(case)
    assert executor.executed_sql == ["SELECT COUNT(*) FROM users;"]


def test_sqlite_executor_rejects_destructive_sql(tmp_path):
    db_path = tmp_path / "test.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    conn.close()

    executor = SQLiteExecutor(db_path)
    with pytest.raises(ValueError, match="Destructive SQL"):
        executor.execute("DROP TABLE users;")
    executor.close()


def test_sqlite_executor_thread_safety_count_vs_alias(tmp_path):
    db_path = _make_sqlite_db(tmp_path)
    executor = SQLiteExecutor(db_path)

    expected_sql = "SELECT COUNT(*) FROM customers"
    generated_sql = "SELECT COUNT(*) AS total_customers FROM customers;"

    expected_rows = _run_sql_on_executor_in_thread(executor, expected_sql)
    actual_rows = _run_sql_on_executor_in_thread(executor, generated_sql)

    assert expected_rows == [(8,)]
    assert actual_rows == [(8,)]

    assert TextToSQLEvaluator._results_equal(actual_rows, expected_rows) is True

    executor.close()


def test_sqlite_executor_no_cross_thread_sqlite_error(tmp_path):
    db_path = _make_sqlite_db(tmp_path)
    executor = SQLiteExecutor(db_path)

    sql = "SELECT COUNT(*) AS total_customers FROM customers;"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_run_sql_on_executor_in_thread, executor, sql) for _ in range(8)]
        rows = [f.result() for f in futures]

    for row in rows:
        assert row == [(8,)]

    executor.close()


def test_sqlite_executor_empty_statement_rejected_in_thread(tmp_path):
    db_path = _make_sqlite_db(tmp_path)
    executor = SQLiteExecutor(db_path)

    def target():
        try:
            executor.execute("")
        except ValueError:
            return
        raise AssertionError("expected empty SQL to be rejected")

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    executor.close()


def test_sqlite_executor_allows_read_only(tmp_path):
    db_path = tmp_path / "test.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    conn.close()

    executor = SQLiteExecutor(db_path)
    result = executor.execute("SELECT COUNT(*) FROM users;")
    assert result == [(1,)]
    executor.close()
