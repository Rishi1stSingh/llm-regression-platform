from src.metrics.base import MetricResult
from src.metrics.engine import MetricsEngine
from src.models import EvaluationRun, CaseResult


class SingleCalculator:
    def compute(self, run: EvaluationRun) -> MetricResult:
        return MetricResult(name="single", value=0.5)


class MultiCalculator:
    def compute(self, run: EvaluationRun) -> list[MetricResult]:
        return [MetricResult(name="m1", value=0.2), MetricResult(name="m2", value=0.8)]


def make_run() -> EvaluationRun:
    cases = [
        CaseResult("1", "a", "x", "x", True),
        CaseResult("2", "b", "y", "wrong", False),
    ]
    run = EvaluationRun("vtest", "fake", cases)
    run.finalize_score()
    return run


def test_metrics_engine_single_and_multi():
    run = make_run()
    engine = MetricsEngine([SingleCalculator(), MultiCalculator()])
    results = engine.compute(run)
    names = [r.name for r in results]
    assert len(results) == 3
    assert "single" in names and "m1" in names and "m2" in names
    # check values preserved
    mapping = {r.name: r.value for r in results}
    assert mapping["single"] == 0.5
    assert mapping["m1"] == 0.2
    assert mapping["m2"] == 0.8
