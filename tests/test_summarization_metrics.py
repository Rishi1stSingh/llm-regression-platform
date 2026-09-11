from __future__ import annotations

from src.metrics.engine import MetricsEngine
from src.metrics.summarization import (
    Rouge1Calculator,
    Rouge2Calculator,
    RougeLCalculator,
    LLMJudgeCalculator,
)
from src.models import CaseResult, EvaluationRun


def make_run(pairs: list[tuple[str, str]], evaluator: str = "summarization", strategy: str = "semantic") -> EvaluationRun:
    cases = [CaseResult(str(i + 1), "doc", ref, gen, True) for i, (ref, gen) in enumerate(pairs)]
    run = EvaluationRun("v", evaluator, cases, strategy=strategy)
    run.finalize_score()
    return run


def test_rouge_perfect_match():
    run = make_run([("the quick brown fox", "the quick brown fox")])
    r1 = Rouge1Calculator().compute(run)
    r2 = Rouge2Calculator().compute(run)
    rl = RougeLCalculator().compute(run)
    assert r1.value == 1.0
    assert r2.value == 1.0
    assert rl.value == 1.0


def test_rouge_partial_overlap():
    run = make_run([("the quick brown fox jumps", "the quick brown fox")])
    r1 = Rouge1Calculator().compute(run)
    assert 0.0 < r1.value < 1.0


def test_rouge_completely_different():
    run = make_run([("the quick brown fox", "completely unrelated content here")])
    r1 = Rouge1Calculator().compute(run)
    assert r1.value == 0.0


def test_rouge_empty_reference():
    run = make_run([("", "some summary")])
    r1 = Rouge1Calculator().compute(run)
    assert r1.value == 0.0


def test_rouge_empty_candidate():
    run = make_run([("reference summary", "")])
    r1 = Rouge1Calculator().compute(run)
    assert r1.value == 0.0


def test_rouge_empty_run():
    run = make_run([])
    r1 = Rouge1Calculator().compute(run)
    assert r1.value == 0.0


def test_rouge_deterministic():
    run = make_run([("the quick brown fox", "the quick brown fox jumps")])
    r1a = Rouge1Calculator().compute(run)
    r1b = Rouge1Calculator().compute(run)
    assert r1a.value == r1b.value


def test_llm_judge_metrics():
    cases = [
        CaseResult("1", "doc", "ref", "gen", True, details={"faithfulness": 0.9, "overall": 0.85}),
        CaseResult("2", "doc", "ref", "gen", True, details={"faithfulness": 0.8, "overall": 0.75}),
    ]
    run = EvaluationRun("v", "summarization", cases, strategy="llm_judge")
    run.finalize_score()
    results = LLMJudgeCalculator().compute(run)
    by_name = {r.name: r.value for r in results}
    assert abs(by_name["faithfulness"] - 0.85) < 1e-6
    assert abs(by_name["overall"] - 0.80) < 1e-6


def test_llm_judge_missing_details():
    cases = [CaseResult("1", "doc", "ref", "gen", True)]
    run = EvaluationRun("v", "summarization", cases, strategy="llm_judge")
    run.finalize_score()
    results = LLMJudgeCalculator().compute(run)
    by_name = {r.name: r.value for r in results}
    assert by_name["faithfulness"] == 0.0
    assert by_name["overall"] == 0.0


def test_auto_select_summarization_semantic():
    run = make_run([("ref", "gen")], strategy="semantic")
    engine = MetricsEngine()
    results = engine.compute(run)
    names = {r.name for r in results}
    assert {"rouge_1", "rouge_2", "rouge_l"}.issubset(names)


def test_auto_select_summarization_llm_judge():
    cases = [CaseResult("1", "doc", "ref", "gen", True, details={"overall": 0.9})]
    run = EvaluationRun("v", "summarization", cases, strategy="llm_judge")
    run.finalize_score()
    engine = MetricsEngine()
    results = engine.compute(run)
    names = {r.name for r in results}
    assert {"faithfulness", "relevance", "completeness", "coherence", "conciseness", "overall"}.issubset(names)