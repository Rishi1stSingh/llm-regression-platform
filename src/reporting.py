from __future__ import annotations

from html import escape
from pathlib import Path

from src.models import EvaluationRun
from typing import Iterable
from src.metrics.base import MetricResult


def generate_html_report(run: EvaluationRun, output_dir: Path, metrics: Iterable[MetricResult] | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"evaluation-{run.run_id}.html"
    delta = "—" if run.delta is None else f"{run.delta:+.1%}"
    baseline = "—" if run.baseline_score is None else f"{run.baseline_score:.1%}"
    metrics = list(metrics or [])
    metrics_rows = ""
    if metrics:
        metrics_rows = "".join(f"<tr><th style='text-align:left;padding:6px'>{escape(m.name.capitalize())}</th><td style='padding:6px'>{m.value:.2%}</td></tr>" for m in metrics)
    # Metric delta rows (baseline vs current)
    delta_rows = ""
    if run.metric_deltas:
        delta_rows = "".join(
            f"<tr><td>{escape(d.name)}</td><td>{'—' if d.baseline is None else f'{d.baseline:.2%}'}</td>"
            f"<td>{'—' if d.current is None else f'{d.current:.2%}'}</td>"
            f"<td>{'—' if d.delta is None else f'{d.delta:+.2%}'}</td>"
            f"<td>{escape(d.status.replace('_', ' ').upper())}</td></tr>"
            for d in run.metric_deltas
        )
    # Metric regression check rows
    check_rows = ""
    if run.metric_regression_checks:
        check_rows = "".join(
            f"<tr><td>{escape(c.name)}</td><td>{'—' if c.baseline is None else f'{c.baseline:.2%}'}</td>"
            f"<td>{'—' if c.current is None else f'{c.current:.2%}'}</td>"
            f"<td>{'—' if c.delta is None else f'{c.delta:+.2%}'}</td>"
            f"<td>{'—' if c.max_drop is None else f'{c.max_drop:.2%}'}</td>"
            f"<td class='{'passed' if c.passed else 'failed'}'>{'PASS' if c.passed else 'FAIL'}</td>"
            f"<td>{escape(c.message)}</td></tr>"
            for c in run.metric_regression_checks
        )
    rows = "".join(
        f"<tr class='{case.change or ''}'><td>{escape(case.case_id)}</td><td>{'PASS' if case.baseline_passed else ('FAIL' if case.baseline_passed is not None else '—')}</td>"
        f"<td>{'PASS' if case.passed else 'FAIL'}</td><td>{escape((case.change or 'new').replace('_', ' ').upper())}</td>"
        f"<td>{escape(case.input)}</td><td>{escape(case.expected)}</td><td>{escape(case.baseline_actual or '—')}</td><td>{escape(case.actual)}</td></tr>"
        for case in run.cases)
    path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>LLM Evaluation Report</title>
    <style>body{{font:15px system-ui;margin:40px;color:#172033}}.card{{padding:18px;border:1px solid #dbe2ea;border-radius:10px;margin:18px 0}}.failed{{color:#b42318}}.passed{{color:#027a48}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #e5e7eb}}th{{background:#f8fafc}}tr.regression{{background:#fff1f1}}tr.improvement{{background:#ecfdf3}}</style></head>
    <body><h1>LLM Evaluation Report</h1><div class='card'><h2 class='{run.status}'>Status: {run.status.upper()}</h2>
    <p><b>Version:</b> {escape(run.version)} &nbsp; <b>Evaluator:</b> {escape(run.evaluator)} &nbsp; <b>Run:</b> {escape(run.run_id)}</p>
    {('<p><b>Strategy:</b> ' + escape(run.strategy) + '</p>') if run.strategy else ''}
    <p><b>Baseline score:</b> {baseline} &nbsp; <b>Candidate score:</b> {run.score:.1%} &nbsp; <b>Delta:</b> {delta}</p>
    <p>Total: {len(run.cases)} | Passed: {run.passed_cases} | Failed: {run.failed_cases} | Regressions: {run.regressions} | Improvements: {run.improvements}</p></div>
    {('<div class="card"><h2>Evaluation Metrics</h2><table>' + metrics_rows + '</table></div>') if metrics_rows else ''}
    {('<div class="card"><h2>Metric Deltas</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th><th>Status</th></tr></thead><tbody>' + delta_rows + '</tbody></table></div>') if delta_rows else ''}
    {('<div class="card"><h2>Metric Regression Checks</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th><th>Max Drop</th><th>Status</th><th>Message</th></tr></thead><tbody>' + check_rows + '</tbody></table></div>') if check_rows else ''}
    <h2>Case details</h2><table><thead><tr><th>ID</th><th>Baseline</th><th>New</th><th>Change</th><th>Input</th><th>Expected</th><th>Baseline output</th><th>New output</th></tr></thead><tbody>{rows}</tbody></table></body></html>""", encoding="utf-8")
    return path