from abc import ABC, abstractmethod

from src.models import CaseResult, GoldenCase


class BaseEvaluator(ABC):
    """Task-specific implementations normalize output into CaseResult."""

    name: str

    @abstractmethod
    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        raise NotImplementedError
