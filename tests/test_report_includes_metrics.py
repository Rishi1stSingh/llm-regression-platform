from pathlib import Path

from src.metrics.base import MetricResult
from src.models import EvaluationRun, CaseResult
from src.reporting import generate_html_report


def test_report_contains_metrics(tmp_path: Path):
    cases = [CaseResult("1", "a", "x", "x", True), CaseResult("2", "b", "y", "wrong", False)]
    run = EvaluationRun("vtest", "classification", cases)
    run.finalize_score()
    metrics = [MetricResult("accuracy", 0.5), MetricResult("precision", 0.4), MetricResult("recall", 0.5), MetricResult("f1", 0.45)]
    report = generate_html_report(run, tmp_path, metrics=metrics)
    text = report.read_text(encoding="utf-8")
    assert "Evaluation Metrics" in text
    assert "Accuracy" in text
    assert "Precision" in text
    assert "Recall" in text
    assert "F1" in text
    # ensure other report content remains
    assert "Case details" in text
    assert "Total:" in text or "Passed:" in text
