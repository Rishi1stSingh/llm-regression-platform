import pytest

from src.evaluators.registry import resolve_evaluator
from src.llm.client import MockLLMClient
from src.evaluators.engine import EvaluationEngine
from src.models import GoldenCase


def test_classification_resolves():
    client = MockLLMClient()
    evaluator = resolve_evaluator("classification", client, "prompt")
    # should be usable by the engine and evaluate a simple case
    engine = EvaluationEngine(evaluator)
    cases = [GoldenCase("1", "The app crashes.", "technical")]
    run = engine.run(cases, "v-test")
    assert run.evaluator == "classification"
    assert run.version == "v-test"
    assert len(run.cases) == 1


def test_unknown_evaluator_raises():
    with pytest.raises(ValueError) as exc:
        resolve_evaluator("no-such-evaluator")
    assert "Unknown evaluator: no-such-evaluator" in str(exc.value)
