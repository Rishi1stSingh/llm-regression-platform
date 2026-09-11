from __future__ import annotations

from src.llm.client import LLMClient
from src.models import CaseResult, GoldenCase
from src.sql_executor import SQLExecutor
from .base import BaseEvaluator


class TextToSQLEvaluator(BaseEvaluator):
    """Generate SQL from a natural-language question, execute it, and compare results.

    The expected result is stored in `case.expected_result` (a list of tuples).
    If `expected_result` is not provided, the evaluator falls back to comparing
    the generated SQL string to `case.expected`.
    """

    name = "text_to_sql"

    def __init__(self, client: LLMClient, prompt: str, executor: SQLExecutor) -> None:
        self.client = client
        self.prompt = prompt
        self.executor = executor

    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        generated_sql = self.client.generate_sql(self.prompt, case.input).strip()
        try:
            result = self.executor.execute(generated_sql)
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                actual=generated_sql,
                passed=False,
                reason=f"SQL execution failed: {exc}",
            )

        expected_result = case.expected_result
        if expected_result is None:
            # Fallback: compare SQL strings (secondary diagnostic)
            passed = generated_sql.strip().rstrip(";").lower() == case.expected.strip().rstrip(";").lower()
            reason = "SQL string match" if passed else "SQL string mismatch"
        else:
            passed = self._results_equal(result, expected_result)
            reason = "Result set match" if passed else "Result set mismatch"

        return CaseResult(
            case_id=case.id,
            input=case.input,
            expected=case.expected,
            actual=generated_sql,
            passed=passed,
            reason=reason,
        )

    @staticmethod
    def _results_equal(actual: list, expected: list) -> bool:
        """Compare two result sets as sets of tuples (order-independent).

        Rows may arrive as tuples (from SQLite) or as plain lists (when the
        expected result is loaded from a JSON array), so both sides are
        normalized to hashable tuples before comparison.
        """
        def _normalize(rows: list) -> set[tuple]:
            normalized: set[tuple] = set()
            for row in rows:
                if isinstance(row, (list, tuple)):
                    normalized.add(tuple(row))
                else:
                    normalized.add((row,))
            return normalized

        return _normalize(actual) == _normalize(expected)