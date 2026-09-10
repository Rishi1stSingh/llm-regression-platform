# LLM Regression Detection Platform

A small, end-to-end LLMOps prototype for detecting quality regressions in a
customer-support ticket classifier. It evaluates versioned prompts against a
golden dataset, stores runs in SQLite, compares a candidate to a baseline,
generates an HTML report, optionally sends a Slack alert, and returns a failing
exit code when a regression exceeds the configured threshold.

## What is included

- NVIDIA NIM / OpenAI-compatible LLM integration
- 32-case, version-controlled golden dataset
- Pluggable `BaseEvaluator` interface and classification evaluator
- SQLite run and case-result history
- Baseline comparison and case-level regression detection
- Standalone HTML reports
- Optional Slack incoming-webhook notification
- GitHub Actions workflow and Dockerfile

## Quick start

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `NVIDIA_API_KEY` in `.env`, then establish the baseline:

```powershell
python main.py run --version v1
```

Run the candidate prompt against the latest successful `v1` run:

```powershell
python main.py run --version v2 --baseline-version v1
```

The command prints the report location and exits with code `1` if the score
drop exceeds `config.json`'s `max_score_drop` (or new case regressions are
configured to fail the run).

For an offline smoke test that makes no API calls, append `--mock`.

## Environment variables

| Name | Required | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Yes for live evaluation | NVIDIA API key |
| `NVIDIA_BASE_URL` | No | Defaults to `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_MODEL` | No | Defaults to `meta/llama-3.1-8b-instruct` |
| `SLACK_WEBHOOK_URL` | No | Incoming webhook for run notifications |

## Project layout

`src/evaluators` contains task-specific code. Everything after the evaluator
uses standardized models, making the comparison, storage, reporting and
notification layers reusable for future summarization, RAG, and Text-to-SQL
evaluators.
