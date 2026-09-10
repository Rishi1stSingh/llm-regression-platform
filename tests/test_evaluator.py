from src.evaluators.classification import ClassificationEvaluator
from src.llm.client import MockLLMClient
from src.models import GoldenCase


def test_mock_classifier_evaluates_a_case():
    prompt = "technical: crashes, errors, uploads, speed, notifications, exports"
    result = ClassificationEvaluator(MockLLMClient(), prompt).evaluate_case(
        GoldenCase("1", "The app crashes on startup.", "technical"))
    assert result.passed is True
