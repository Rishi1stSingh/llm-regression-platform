from src.llm.client import LLMClient
from src.models import CaseResult, GoldenCase
from .base import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    name = "classification"
    labels = {"billing", "account", "technical", "general"}

    def __init__(self, client: LLMClient, prompt: str) -> None:
        self.client = client
        self.prompt = prompt

    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        actual = self.client.classify(self.prompt, case.input).strip().lower()
        if actual not in self.labels:
            return CaseResult(case.id, case.input, case.expected, actual, False, "Invalid LLM label")
        return CaseResult(case.id, case.input, case.expected, actual, actual == case.expected)
