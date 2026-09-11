from __future__ import annotations

import re
from collections import Counter

from src.metrics.base import MetricResult
from src.models import EvaluationRun


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"\b[\w']+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_precision_recall_f1(reference: str, candidate: str, n: int) -> tuple[float, float, float]:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0, 0.0, 0.0
    ref_ngrams = _ngrams(ref_tokens, n)
    cand_ngrams = _ngrams(cand_tokens, n)
    overlap = sum((ref_ngrams & cand_ngrams).values())
    precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class Rouge1Calculator:
    name = "rouge_1"

    def compute(self, run: EvaluationRun) -> MetricResult:
        if not run.cases:
            return MetricResult(name=self.name, value=0.0)
        scores = []
        for case in run.cases:
            reference = case.expected
            candidate = case.actual
            _, _, f1 = _rouge_n_precision_recall_f1(reference, candidate, 1)
            scores.append(f1)
        return MetricResult(name=self.name, value=sum(scores) / len(scores))


class Rouge2Calculator:
    name = "rouge_2"

    def compute(self, run: EvaluationRun) -> MetricResult:
        if not run.cases:
            return MetricResult(name=self.name, value=0.0)
        scores = []
        for case in run.cases:
            reference = case.expected
            candidate = case.actual
            _, _, f1 = _rouge_n_precision_recall_f1(reference, candidate, 2)
            scores.append(f1)
        return MetricResult(name=self.name, value=sum(scores) / len(scores))


class RougeLCalculator:
    name = "rouge_l"

    def compute(self, run: EvaluationRun) -> MetricResult:
        if not run.cases:
            return MetricResult(name=self.name, value=0.0)
        scores = []
        for case in run.cases:
            reference = case.expected
            candidate = case.actual
            scores.append(_rouge_l_f1(reference, candidate))
        return MetricResult(name=self.name, value=sum(scores) / len(scores))


class LLMJudgeCalculator:
    """Compute average judge scores from case details.

    Each case stores its judge scores in `case.details` (e.g. {"faithfulness": 0.95, ...}).
    This calculator produces one MetricResult per dimension.
    """

    name = "llm_judge"
    dimensions = ("faithfulness", "relevance", "completeness", "coherence", "conciseness", "overall")

    def compute(self, run: EvaluationRun) -> list[MetricResult]:
        results: list[MetricResult] = []
        for dim in self.dimensions:
            values = []
            for case in run.cases:
                if case.details and dim in case.details:
                    values.append(float(case.details[dim]))
            avg = sum(values) / len(values) if values else 0.0
            results.append(MetricResult(name=dim, value=avg))
        return results