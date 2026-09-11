from src.evaluators.base import BaseEvaluator
from src.models import GoldenCase, CaseResult
from src.evaluators.engine import EvaluationEngine


class FakeCustomEvaluator(BaseEvaluator):
    name = "fake_custom"

    def __init__(self, multiplier: int = 1) -> None:
        self.multiplier = multiplier
        self.calls = 0

    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        # deterministic logic: pass if case id as int % multiplier != 0
        self.calls += 1
        try:
            val = int(case.id)
        except Exception:
            val = 0
        passed = (val % self.multiplier) != 0
        actual = case.expected if passed else f"wrong-{case.expected}"
        return CaseResult(case.id, case.input, case.expected, actual, passed)


class NotAnEvaluator:
    """Deliberately does NOT inherit BaseEvaluator (for registry validation tests)."""
    pass


def test_custom_evaluator_plugs_into_engine():
    cases = [GoldenCase("1", "a", "x"), GoldenCase("2", "b", "y"), GoldenCase("3", "c", "z")]
    evaluator = FakeCustomEvaluator(multiplier=3)
    engine = EvaluationEngine(evaluator)
    run = engine.run(cases, "v-custom")
    # fake evaluator should have been called once per case
    assert evaluator.calls == 3
    # evaluator name propagated
    assert run.evaluator == "fake_custom"
    # score calculation: only ids not divisible by 3 pass -> ids 1 and 2 pass => 2/3
    assert run.score == 2 / 3
    assert len(run.cases) == 3
