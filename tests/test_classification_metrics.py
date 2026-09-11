from src.metrics.classification import PrecisionCalculator, RecallCalculator, F1Calculator
from src.metrics.engine import MetricsEngine
from src.models import EvaluationRun, CaseResult


def make_run(pairs: list[tuple[str, str]]) -> EvaluationRun:
    cases = [CaseResult(str(i + 1), "in", exp, act, exp == act) for i, (exp, act) in enumerate(pairs)]
    run = EvaluationRun("v", "classification", cases)
    run.finalize_score()
    return run


def test_perfect_predictions():
    pairs = [("billing", "billing"), ("account", "account"), ("technical", "technical"), ("general", "general")]
    run = make_run(pairs)
    p = PrecisionCalculator().compute(run)
    r = RecallCalculator().compute(run)
    f = F1Calculator().compute(run)
    assert p.value == 1.0
    assert r.value == 1.0
    assert f.value == 1.0


def test_completely_wrong_predictions():
    pairs = [("billing", "account"), ("account", "billing"), ("technical", "general"), ("general", "technical")]
    run = make_run(pairs)
    p = PrecisionCalculator().compute(run)
    r = RecallCalculator().compute(run)
    f = F1Calculator().compute(run)
    assert p.value == 0.0
    assert r.value == 0.0
    assert f.value == 0.0


def test_missing_class_in_predictions():
    # no predictions for 'general' (actual never 'general')
    pairs = [("billing", "billing"), ("account", "account"), ("technical", "technical"), ("general", "billing")]
    run = make_run(pairs)
    p = PrecisionCalculator().compute(run)
    r = RecallCalculator().compute(run)
    f = F1Calculator().compute(run)
    # per-class values should be computed; macro averages should be between 0 and 1
    assert 0.0 <= p.value <= 1.0
    assert 0.0 <= r.value <= 1.0
    assert 0.0 <= f.value <= 1.0
    assert "general" in p.details


def test_mixed_predictions_manual_check():
    # Construct a small set where we can compute per-class metrics manually
    pairs = [
        ("billing", "billing"),  # billing TP
        ("billing", "account"),  # billing FN, account FP
        ("account", "account"),  # account TP
        ("technical", "technical"),  # technical TP
    ]
    run = make_run(pairs)
    p = PrecisionCalculator().compute(run)
    r = RecallCalculator().compute(run)
    f = F1Calculator().compute(run)
    # per-class expectations:
    # billing: TP=1, FP=0, FN=1 -> precision=1/(1+0)=1.0, recall=1/(1+1)=0.5, f1=2*1*0.5/(1+0.5)=0.666...
    billing_precision = p.details["billing"]
    billing_recall = r.details["billing"]
    billing_f1 = f.details["billing"]
    assert abs(billing_precision - 1.0) < 1e-6
    assert abs(billing_recall - 0.5) < 1e-6
    assert abs(billing_f1 - (2 * 1.0 * 0.5 / (1.0 + 0.5))) < 1e-6


def test_empty_run_metrics_zero():
    run = make_run([])
    p = PrecisionCalculator().compute(run)
    r = RecallCalculator().compute(run)
    f = F1Calculator().compute(run)
    assert p.value == 0.0
    assert r.value == 0.0
    assert f.value == 0.0


def test_metrics_engine_returns_all():
    pairs = [("billing", "billing"), ("account", "account"), ("technical", "technical"), ("general", "general")]
    run = make_run(pairs)
    engine = MetricsEngine([PrecisionCalculator(), RecallCalculator(), F1Calculator()])
    results = engine.compute(run)
    names = {r.name for r in results}
    assert {"precision", "recall", "f1"}.issubset(names)
