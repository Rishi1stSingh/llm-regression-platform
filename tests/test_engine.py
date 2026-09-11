from src.evaluators.engine import EvaluationEngine
from src.evaluators.base import BaseEvaluator
from src.models import GoldenCase, CaseResult


class FakeEvaluator(BaseEvaluator):
    name = "fake"

    def __init__(self, pass_ids: set[str] | None = None) -> None:
        self.pass_ids = pass_ids or set()
        self.call_count = 0

    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        self.call_count += 1
        passed = case.id in self.pass_ids
        actual = case.expected if passed else f"wrong-{case.expected}"
        return CaseResult(case.id, case.input, case.expected, actual, passed)


def test_all_pass_three_cases():
    cases = [GoldenCase("1", "a", "x"), GoldenCase("2", "b", "y"), GoldenCase("3", "c", "z")]
    evaluator = FakeEvaluator(pass_ids={"1", "2", "3"})
    engine = EvaluationEngine(evaluator)
    run = engine.run(cases, "v1")
    assert run.score == 1.0
    assert run.passed_cases == 3
    assert evaluator.call_count == 3
    assert run.evaluator == evaluator.name
    assert run.version == "v1"


def test_three_pass_one_fail_four_cases():
    cases = [GoldenCase("1", "a", "x"), GoldenCase("2", "b", "y"), GoldenCase("3", "c", "z"), GoldenCase("4", "d", "w")]
    evaluator = FakeEvaluator(pass_ids={"1", "2", "3"})
    engine = EvaluationEngine(evaluator)
    run = engine.run(cases, "v2")
    assert run.score == 0.75
    assert run.passed_cases == 3
    assert run.failed_cases == 1
    assert evaluator.call_count == 4
    assert run.evaluator == evaluator.name
    assert run.version == "v2"


def test_empty_dataset():
    cases: list[GoldenCase] = []
    evaluator = FakeEvaluator()
    engine = EvaluationEngine(evaluator)
    run = engine.run(cases, "v-empty")
    assert run.score == 0.0
    assert run.passed_cases == 0
    assert evaluator.call_count == 0
    assert run.evaluator == evaluator.name
    assert run.version == "v-empty"
