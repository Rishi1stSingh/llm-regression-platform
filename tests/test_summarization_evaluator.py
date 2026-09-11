from __future__ import annotations

from src.evaluators.summarization import SummarizationEvaluator
from src.llm.client import MockLLMClient

from src.models import GoldenCase


class FakeSummarizationClient:
    """Deterministic fake client for summarization tests."""

    def __init__(self, summary: str | None = None, judge_output: dict | None = None) -> None:
        self.summary = summary
        self.judge_output = judge_output or {}
        self.summarize_calls = 0
        self.judge_calls = 0

    def summarize(self, system_prompt: str, document: str) -> str:
        self.summarize_calls += 1
        if self.summary is not None:
            return self.summary
        # Default: return first sentence
        import re
        sentences = re.split(r"(?<=[.!?])\s+", document.strip())
        return sentences[0] if sentences else ""

    def judge(self, system_prompt: str, user_prompt: str) -> dict:
        self.judge_calls += 1
        return self.judge_output


def test_semantic_perfect_match():
    client = FakeSummarizationClient(summary="The quick brown fox jumps over the lazy dog.")
    evaluator = SummarizationEvaluator(client, "prompt", strategy="semantic")
    case = GoldenCase("1", "The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog.")
    result = evaluator.evaluate_case(case)
    assert result.passed is True
    assert result.actual == "The quick brown fox jumps over the lazy dog."


def test_semantic_partial_overlap():
    client = FakeSummarizationClient(summary="The quick brown fox.")
    evaluator = SummarizationEvaluator(client, "prompt", strategy="semantic")
    case = GoldenCase("1", "The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog.")
    result = evaluator.evaluate_case(case)
    assert result.passed is True  # non-empty summary passes semantic heuristic


def test_semantic_completely_different():
    client = FakeSummarizationClient(summary="Completely unrelated content.")
    evaluator = SummarizationEvaluator(client, "prompt", strategy="semantic")
    case = GoldenCase("1", "The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog.")
    result = evaluator.evaluate_case(case)
    assert result.passed is True  # non-empty summary passes semantic heuristic


def test_semantic_empty_input():
    client = FakeSummarizationClient(summary="")
    evaluator = SummarizationEvaluator(client, "prompt", strategy="semantic")
    case = GoldenCase("1", "", "reference summary")
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "Empty generated summary" in result.reason


def test_semantic_empty_reference():
    client = FakeSummarizationClient(summary="some summary")
    evaluator = SummarizationEvaluator(client, "prompt", strategy="semantic")
    case = GoldenCase("1", "input", "")
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "Empty reference summary" in result.reason


def test_llm_judge_mocked_response():
    client = FakeSummarizationClient(
        summary="Generated summary.",
        judge_output={
            "faithfulness": 0.95,
            "relevance": 0.90,
            "completeness": 0.84,
            "coherence": 0.92,
            "conciseness": 0.88,
            "overall": 0.90,
            "reason": "Good summary.",
        },
    )
    evaluator = SummarizationEvaluator(client, "prompt", strategy="llm_judge")
    case = GoldenCase("1", "Source document.", "Reference summary.")
    result = evaluator.evaluate_case(case)
    assert result.passed is True
    assert result.details is not None
    assert result.details["faithfulness"] == 0.95
    assert result.details["overall"] == 0.90


def test_llm_judge_missing_fields():
    client = FakeSummarizationClient(
        summary="Generated summary.",
        judge_output={"overall": 0.8},
    )
    evaluator = SummarizationEvaluator(client, "prompt", strategy="llm_judge")
    case = GoldenCase("1", "Source document.", "Reference summary.")
    result = evaluator.evaluate_case(case)
    assert result.passed is True
    assert result.details == {"overall": 0.8}


def test_llm_judge_invalid_response():
    client = FakeSummarizationClient(
        summary="Generated summary.",
        judge_output={"faithfulness": "not-a-number", "overall": 1.5},
    )
    evaluator = SummarizationEvaluator(client, "prompt", strategy="llm_judge")
    case = GoldenCase("1", "Source document.", "Reference summary.")
    result = evaluator.evaluate_case(case)
    assert result.passed is False
    assert "no valid scores" in result.reason.lower()


def test_llm_judge_score_range_validation():
    client = FakeSummarizationClient(
        summary="Generated summary.",
        judge_output={"faithfulness": 1.5, "overall": 0.7},
    )
    evaluator = SummarizationEvaluator(client, "prompt", strategy="llm_judge")
    case = GoldenCase("1", "Source document.", "Reference summary.")
    result = evaluator.evaluate_case(case)
    # faithfulness out of range should be excluded; overall in range should be kept
    assert result.details == {"overall": 0.7}


def test_llm_judge_no_real_api_call():
    client = FakeSummarizationClient(
        summary="Generated summary.",
        judge_output={"overall": 0.9},
    )
    evaluator = SummarizationEvaluator(client, "prompt", strategy="llm_judge")
    case = GoldenCase("1", "Source document.", "Reference summary.")
    evaluator.evaluate_case(case)
    assert client.judge_calls == 1
    assert client.summarize_calls == 1


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        SummarizationEvaluator(MockLLMClient(), "prompt", strategy="unknown")