from __future__ import annotations

from typing import Any

from src.llm.client import LLMClient
from src.models import CaseResult, GoldenCase
from .base import BaseEvaluator

JUDGE_DIMENSIONS = ("faithfulness", "relevance", "completeness", "coherence", "conciseness", "overall")


class SummarizationEvaluator(BaseEvaluator):
    """Generate a summary for a document and evaluate it.

    Strategies:
      - semantic: compare generated summary to reference summary (ROUGE metrics computed by MetricsEngine)
      - llm_judge: use an LLM judge to score faithfulness, relevance, completeness, coherence, conciseness, overall
    """

    name = "summarization"

    def __init__(self, client: LLMClient, prompt: str, strategy: str = "semantic",
                 pass_threshold: float = 0.5) -> None:
        self.client = client
        self.prompt = prompt
        if strategy not in ("semantic", "llm_judge"):
            raise ValueError(f"Unknown summarization strategy: {strategy}")
        self.strategy = strategy
        self.pass_threshold = pass_threshold

    def evaluate_case(self, case: GoldenCase) -> CaseResult:
        generated = self.client.summarize(self.prompt, case.input).strip()
        reference = case.expected.strip()

        if self.strategy == "semantic":
            return self._evaluate_semantic(case, generated, reference)
        return self._evaluate_llm_judge(case, generated, reference)

    def _evaluate_semantic(self, case: GoldenCase, generated: str, reference: str) -> CaseResult:
        if not reference:
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                actual=generated,
                passed=False,
                reason="Empty reference summary",
            )
        if not generated:
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                actual=generated,
                passed=False,
                reason="Empty generated summary",
            )
        # Semantic strategy: pass/fail is determined by the metrics engine (ROUGE scores).
        # Here we use a simple heuristic: pass if the generated summary is non-empty.
        # The actual ROUGE-based pass/fail is computed by the metrics engine.
        passed = True
        return CaseResult(
            case_id=case.id,
            input=case.input,
            expected=case.expected,
            actual=generated,
            passed=passed,
            reason="Summary generated",
        )

    def _evaluate_llm_judge(self, case: GoldenCase, generated: str, reference: str) -> CaseResult:
        if not generated:
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                actual=generated,
                passed=False,
                reason="Empty generated summary",
            )
        user_prompt = (
            f"Source document:\n{case.input}\n\n"
            f"Generated summary:\n{generated}\n"
        )
        if reference:
            user_prompt += f"\nReference summary:\n{reference}\n"
        user_prompt += (
            "\nReturn a JSON object with numeric scores (0.0 to 1.0) for: "
            "faithfulness, relevance, completeness, coherence, conciseness, overall. "
            "Also include a 'reason' string."
        )
        try:
            judge_output = self.client.judge(self.prompt, user_prompt)
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                actual=generated,
                passed=False,
                reason=f"Judge error: {exc}",
            )

        scores = self._extract_scores(judge_output)
        if not scores:
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                actual=generated,
                passed=False,
                reason="Judge returned no valid scores",
            )

        overall = scores.get("overall", 0.0)
        passed = overall >= self.pass_threshold
        reason = str(judge_output.get("reason", "")) if isinstance(judge_output, dict) else ""
        return CaseResult(
            case_id=case.id,
            input=case.input,
            expected=case.expected,
            actual=generated,
            passed=passed,
            reason=reason or f"Overall score: {overall:.2f}",
            details=scores,
        )

    def _extract_scores(self, judge_output: Any) -> dict[str, float]:
        """Extract numeric scores from judge output, validating ranges."""
        if not isinstance(judge_output, dict):
            return {}
        scores: dict[str, float] = {}
        for dim in JUDGE_DIMENSIONS:
            raw = judge_output.get(dim)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 1.0:
                scores[dim] = value
        return scores