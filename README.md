# LLM Regression Detection Platform

A production-ready LLMOps prototype for detecting quality regressions in LLM-powered
classification. The platform evaluates versioned prompts against a golden dataset,
computes multiple metrics (accuracy, precision, recall, F1), compares candidates
against baselines, generates detailed HTML reports, optionally sends Slack alerts,
and returns failing exit codes when regressions exceed configurable thresholds.

## What is included

- **Multi-provider LLM integration**: Groq (async, HTTPX), NVIDIA NIM/OpenAI-compatible, and deterministic mock provider for offline testing
- **Golden dataset**: 100-case, version-controlled dataset covering billing, account, technical, and general support ticket categories
- **Pluggable evaluator architecture**: Extensible `BaseEvaluator` interface supporting classification, summarization, text-to-SQL, and custom evaluators
- **Comprehensive metrics engine**: Auto-selected metrics per evaluator type with extensible registry
- **SQLite run history**: Full persistence of runs, case-level results, and metrics for historical comparison
- **Baseline comparison**: Case-level regression detection comparing current vs. baseline runs
- **Metric regression checks**: Threshold-based regression detection on accuracy, F1, and other metrics
- **Standalone HTML reports**: Detailed interactive reports with pass/fail status, metrics, deltas, and case-level breakdowns
- **Optional Slack notifications**: Incoming webhook integration for run status alerts
- **GitHub Actions CI**: Automated unit testing + real Groq evaluation on every push/PR
- **Docker support**: Containerized runs for portability
- **Rate limiting & retries**: Configurable concurrency, requests-per-minute, and retry logic for API resilience

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Groq API key (free at [groq.com](https://groq.com)) **or** NVIDIA API key

### Setup

```bash
# Clone and enter the repository
git clone https://github.com/Rishi1stSingh/llm-regression-platform.git
cd llm-regression-platform

# Option A: Using uv (recommended)
uv sync

# Option B: Using pip
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# or .venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Copy the example environment file
cp .env.example .env
```

### Configure API keys

Edit `.env` and add your API key(s):

```bash
# At minimum, add your Groq key (used by CI):
GROQ_API_KEY=your-groq-api-key-here

# Optional: NVIDIA NIM key (alternative provider)
NVIDIA_API_KEY=your-nvidia-key-here

# Optional: Slack webhook for notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## CLI Reference

The CLI (`main.py`) is the actual entry point to the evaluation framework. One
invocation loads and validates a golden dataset, evaluates every case through
the selected LLM provider and evaluator, computes metrics, compares against a
baseline, stores the run in SQLite (`artifacts/evaluations.db`), generates an
HTML report (`artifacts/reports/`), and sends a Slack notification unless
disabled.

### General command structure

```bash
uv run python main.py <command> [options]
```

The CLI currently supports a single command: **`run`**. All of its options are
optional except `--version`. Discover the full surface directly from the CLI:

```bash
uv run python main.py --help          # list available commands
uv run python main.py run --help      # show every `run` option
```

### The `run` command

```bash
uv run python main.py run [options]
```

Executes one evaluation:

1. Resolves the golden dataset (`--dataset` type name or file path, else
   `evaluator.type` from `config.json`).
2. Validates every case (`id`, `input`, `expected` non-empty, unique IDs).
3. Evaluates each case through the resolved provider/evaluator using the
   task-specific system prompt for the chosen version (`prompts/<version>.txt`
   for classification, `prompts/summarization_<version>.txt` for summarization,
   `prompts/text_to_sql_<version>.txt` for text-to-SQL).
4. Computes evaluator-specific metrics and metric regression checks.
5. Compares against the baseline when `--baseline-version` is given.
6. Saves the run to SQLite and writes an HTML report.
7. Sends a Slack notification unless `--no-slack`.

Exit codes:

| Code | Meaning |
| ---- | ------- |
| `0`  | Run passed the regression gate |
| `1`  | Run failed the gate (score drop, case regressions, or metric check) |
| `2`  | Invalid arguments or invalid custom dataset (missing file, invalid JSON, duplicate IDs, missing/empty required fields) |

#### Option reference

Populated from the real argparse implementation (see `uv run python main.py run --help`):

| Option | Required | Description | Example |
| ------ | -------- | ----------- | ------- |
| `--version` | Yes (choices: `v1`, `v2`) | Evaluation version; selects the task-specific prompt file for the evaluator (`prompts/<version>.txt`, `prompts/summarization_<version>.txt`, `prompts/text_to_sql_<version>.txt`) | `--version v1` |
| `--baseline-version` | No | Compare against the latest successful run of this version from SQLite | `--baseline-version v1` |
| `--provider` | No (choices: `groq`, `nvidia`, `mock`) | LLM provider override (default: `llm.provider` in `config.json`) | `--provider groq` |
| `--mock` | No (flag) | Force the deterministic offline client | `--mock` |
| `--no-slack` | No (flag) | Skip the Slack notification even if `SLACK_WEBHOOK_URL` is set | `--no-slack` |
| `--async` | No (flag) | Evaluate cases concurrently | `--async` |
| `--concurrency` | No (default: `5`) | Max concurrent evaluations when using `--async` | `--concurrency 10` |
| `--cases` | No | Limit evaluation to the first N golden-dataset cases | `--cases 10` |
| `--dataset` | No | Built-in dataset type (`classification`, `summary`, `sql`) **or** a path to your own golden-dataset JSON file — a path takes precedence (default: `classification`) | `--dataset ./data/golden/my_cases.json` |
| `--evaluator` | No (choices: `classification`, `sql`, `summary`) | Evaluator to use when `--dataset` points to a custom file (default: `evaluator.type` in `config.json`; ignored when `--dataset` is a type name) | `--evaluator summary` |
| `-h`, `--help` | No | Show help for the CLI or the `run` subcommand | `run --help` |

### Practical command examples

#### Run a normal evaluation

```bash
uv run python main.py run --version v1 --provider groq
```

Runs evaluation version `v1` on the built-in classification golden dataset
(`data/golden/classification_cases.json`), prints metrics under
`Evaluation Results`, saves the run to SQLite, and generates an HTML report.

#### Run with a custom golden dataset

```bash
uv run python main.py run \
  --version v1 \
  --provider groq \
  --dataset ./data/golden/my_cases.json
```

Evaluates your own dataset file, validated with the same rules as the built-in
datasets. Relative and absolute paths both work (see
[Providing your own golden dataset](#providing-your-own-golden-dataset-no-code-changes-required)).

#### Run mock evaluation

```bash
uv run python main.py run \
  --version v1 \
  --mock \
  --no-slack
```

Fully offline, deterministic run — no API keys or network required. Use it to
verify the setup before spending real provider quota.

#### Run baseline comparison

```bash
uv run python main.py run \
  --version v2 \
  --baseline-version v1 \
  --provider groq
```

Runs version `v2` and compares the score and individual case outcomes against
the latest successful `v1` run in SQLite. The run exits `1` if:

- The score drop exceeds `max_score_drop` from `config.json`
- New case regressions are found while `fail_on_new_regressions` is enabled
- Any metric regression check fails (e.g. accuracy or F1 drop exceeds `max_drop`)

#### Establish a baseline

```bash
uv run python main.py run --version v1 --provider groq
```

The first successful run of a version automatically becomes the baseline for
future comparisons of the same version.

#### Limit evaluation cases (quick smoke run)

```bash
uv run python main.py run --version v1 --cases 10
```

Evaluates only the first 10 cases of the resolved golden dataset.

#### Run evaluations concurrently

```bash
uv run python main.py run --version v1 --provider mock --async --concurrency 10
```

Evaluates cases concurrently using async HTTP requests; `--concurrency`
controls the maximum number of simultaneous evaluations (default: 5). Async
mode requires an async-compatible provider (groq or mock).

#### Disable Slack notifications

```bash
uv run python main.py run --version v1 --no-slack
```

Skips sending the Slack notification even if `SLACK_WEBHOOK_URL` is configured.

### Common CLI Workflows

The workflows a developer runs most often, each with the exact command:

**1. Local mock evaluation (offline, no API key)**

```bash
uv run python main.py run --version v1 --mock --no-slack
```

**2. Local real-provider evaluation**

```bash
uv run python main.py run --version v1 --provider groq
```

Requires `GROQ_API_KEY` in your `.env` file. For the alternative NVIDIA
provider, use `--provider nvidia` (requires `NVIDIA_API_KEY`).

**3. Evaluation using a custom golden dataset**

```bash
uv run python main.py run --version v1 --provider groq --dataset ./data/golden/my_cases.json
```

See [Providing your own golden dataset](#providing-your-own-golden-dataset-no-code-changes-required) below.

**4. Version evaluation**

```bash
uv run python main.py run --version v2 --provider groq
```

Each version selects its own task-specific prompt files
(`prompts/v1.txt`, `prompts/summarization_v1.txt`, `prompts/text_to_sql_v1.txt`).

**5. Baseline/regression comparison**

```bash
uv run python main.py run --version v2 --baseline-version v1 --provider groq
```

**6. Evaluation without Slack**

```bash
uv run python main.py run --version v1 --no-slack
```

**7. Quick smoke run (10 cases, offline)**

```bash
uv run python main.py run --version v1 --mock --no-slack --cases 10
```

### Providing your own golden dataset (no code changes required)

The framework is a reusable evaluation engine: your golden dataset belongs to
your project, not to this repository. Point `--dataset` at a JSON file and the
framework validates, evaluates, stores, and reports on it using the exact same
pipeline as the built-in datasets. You never need to edit `GOLDEN_DATASETS` in
`src/golden.py` or import any Python function.

To use it:

1. Create your golden dataset as a JSON array of case objects.
2. Follow the required JSON structure below (`id`, `input`, `expected`, all
   non-empty, unique IDs).
3. Run the evaluator with `--dataset ./path/to/your_cases.json`.
4. The framework validates the file (missing file, invalid JSON, duplicate IDs,
   and missing/empty required fields all fail fast with exit code `2`) and then
   evaluates it.

Required JSON structure:

```json
[
  {
    "id": "CASE_001",
    "input": "your input",
    "expected": "expected output"
  }
]
```

Notes:

- `--dataset` also still accepts the built-in dataset types
  (`classification`, `summary`, `sql`). Any other value is treated as a file
  path, which takes precedence over the built-in datasets.
- The evaluator is chosen from `config.json` (`evaluator.type`) unless you
  pass `--evaluator {classification,summary,sql}`. A custom classification-style
  dataset works with the default; use `--evaluator summary` for summarization
  tasks.
- Everything else behaves identically for custom datasets: versioning with
  `--version`/`--baseline-version`, regression comparison, SQLite storage,
  HTML reports, Slack notifications, and the CI pass/fail gate.
- In GitHub Actions, just commit your dataset and reference it:

```yaml
- name: Run LLM evaluation
  run: |
    uv run python main.py run \
      --version v1 \
      --provider groq \
      --dataset ./data/golden/my_cases.json
```

### Help commands

argparse supports help on every level of the CLI. These display the available
commands and options directly from the implementation:

```bash
uv run python main.py --help        # general help: available commands
uv run python main.py run --help    # `run` subcommand: every option, choice, and default
```

Example output of `uv run python main.py --help`:

```text
usage: main.py [-h] {run} ...

Run LLM regression evaluation.

positional arguments:
  {run}

options:
  -h, --help  show this help message and exit
```

### Important distinction: CLI vs GitHub Actions

```text
CLI
→ Used by developers locally and by CI/CD

GitHub Actions
→ Automatically executes the CLI during the CI workflow
```

The CLI is the actual entry point to the evaluation framework. GitHub Actions
is one automated way of running it: `.github/workflows/ci.yml` invokes
`uv run python main.py run ...` on every push/pull request and relies on the
same exit codes as the local regression gate.


## Configuration

All configuration lives in `config.json`:

```json
{
  "llm": {
    "provider": "groq",
    "concurrency": 5,
    "requests_per_minute": 25,
    "max_retries": 3
  },
  "evaluator": {
    "type": "classification"
  },
  "regression": {
    "max_score_drop": 0.02,
    "fail_on_new_regressions": true,
    "metrics": [
      {"name": "accuracy", "max_drop": 0.02},
      {"name": "f1", "max_drop": 0.03}
    ]
  }
}
```

| Field | Description |
|-------|-------------|
| `llm.provider` | Default LLM provider: `groq`, `nvidia`, or `mock` |
| `llm.concurrency` | Max concurrent evaluations in async mode |
| `llm.requests_per_minute` | Rate limiting: max requests per minute |
| `llm.max_retries` | Max retry attempts on API failures |
| `evaluator.type` | Evaluator type: `classification`, `summarization`, `text_to_sql`, `custom` |
| `regression.max_score_drop` | Maximum allowed score drop before failing (0.02 = 2%) |
| `regression.fail_on_new_regressions` | Fail if new case regressions are found |
| `regression.metrics` | List of metric threshold configs with `name` and `max_drop` |

## Environment variables

| Name | Required | Default | Purpose |
|------|----------|---------|---------|
| `GROQ_API_KEY` | If using Groq provider | — | Groq API key |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Groq model identifier |
| `GROQ_BASE_URL` | No | `https://api.groq.com/openai/v1` | Groq API base URL |
| `NVIDIA_API_KEY` | If using NVIDIA provider | — | NVIDIA API key |
| `NVIDIA_MODEL` | No | `deepseek-ai/deepseek-v4-flash-0731` | NVIDIA model identifier |
| `NVIDIA_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | NVIDIA API base URL |
| `SLACK_WEBHOOK_URL` | No | — | Slack incoming webhook for notifications |

## Project layout

```
llm-regression-platform/
├── .github/workflows/
│   ├── ci.yml                 # CI: unit tests + real Groq evaluation
│   └── evaluate.yml           # Legacy evaluation workflow
├── artifacts/                 # Generated artifacts (reports, SQLite DB)
│   ├── reports/               # HTML evaluation reports
│   └── evaluations.db         # SQLite evaluation database
├── data/
│   └── golden/                        # Golden datasets, one per evaluator
│       ├── classification_cases.json  # 100-case classification dataset
│       ├── summary_cases.json         # Summarization golden cases
│       ├── sql_cases.json             # Text-to-SQL golden cases
│       └── sql_schema.sql             # SQLite schema/seed for text-to-SQL
├── prompts/
│   ├── v1.txt, v2.txt                    # Classification system prompts
│   ├── summarization_v1.txt, _v2.txt     # Summarization system prompts
│   └── text_to_sql_v1.txt, _v2.txt       # Text-to-SQL system prompts
├── src/
│   ├── cli.py                 # CLI entry point and argument parsing
│   ├── comparator.py          # Baseline comparison and metric regression checks
│   ├── models.py              # Dataclasses: GoldenCase, CaseResult, EvaluationRun
│   ├── reporting.py           # HTML report generation
│   ├── storage.py             # SQLite repository for runs, cases, metrics
│   ├── sql_executor.py        # Read-only SQL executor for text-to-SQL
│   ├── notifications.py       # Slack notification integration
│   ├── llm/                   # LLM client implementations
│   │   ├── client.py          # Groq, NVIDIA, Mock async/sync clients
│   │   └── __init__.py
│   ├── metrics/               # Metric calculators and registry
│   │   ├── base.py            # MetricResult, MetricCalculator base
│   │   ├── engine.py          # MetricsEngine orchestrator
│   │   ├── registry.py        # Auto-metric selection by evaluator
│   │   └── *.py               # Individual metric calculators
│   └── evaluators/            # Task-specific evaluators
│       ├── base.py            # BaseEvaluator abstract class
│       ├── classification.py  # Support ticket classifier
│       ├── summarization.py   # Text summarization evaluator
│       ├── text_to_sql.py     # Text-to-SQL evaluator with SQL validation
│       ├── custom.py          # Dynamic custom evaluator loading
│       ├── engine.py          # Sync/async evaluation engines
│       ├── registry.py        # Evaluator name resolution
│       └── __init__.py
├── tests/                     # Unit tests
│   ├── test_*.py              # Comprehensive test suite
├── main.py                    # Application entry point
├── config.json                # Configuration
├── .env.example               # Template for environment variables
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata and tool config
└── Dockerfile                 # Container image definition
```

## GitHub Actions CI/CD

The project includes automated CI workflows that run on every push and pull request:

### Workflow: `ci.yml`

1. **`test` job** — Runs the full unit test suite with pytest (Python 3.12, uv)
2. **`evaluate` job** — Runs real Groq evaluation (depends on `test` passing)

The evaluation job:

- Uploads `GROQ_API_KEY` from repository secrets (never logged)
- Runs: `uv run python main.py run --version v1 --provider groq --no-slack`
- Uploads evaluation reports and database as artifacts (always, even on failure)
- Fails the job if evaluation exits non-zero (regression detected)

### Required secrets

Add `GROQ_API_KEY` as a repository secret in GitHub Settings → Secrets and variables → Actions.

### Manual triggering

Workflows can be triggered manually via the `workflow_dispatch` event or by
running the evaluate workflow directly from the GitHub UI.

## Docker

```bash
# Build the container image
docker build -t regressor-platform .

# Run the default evaluation (v1 with the provider from config.json)
docker run --rm -v $(pwd)/artifacts:/app/artifacts regressor-platform

# Run with your API key
docker run --rm --env GROQ_API_KEY=$GROQ_API_KEY -v $(pwd)/artifacts:/app/artifacts regressor-platform python main.py run --version v1 --provider groq
```

## Testing

```bash
# Run all unit tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_comparator.py -v

# Run tests that make no API calls (deterministic)
uv run pytest tests/test_comparator.py tests/test_storage.py tests/test_groq_client.py -v

# Run with timeout (recommended for CI)
uv run pytest tests/ -v --tb=short --timeout=60
```

Tests cover: comparator logic, HTML reporting, SQLite storage, Groq/NVIDIA mock
clients, evaluator registry, metrics engine, SQL executor validation, and
regression detection across all metric types.

## How it works

1. **Load**: Reads `config.json` and `data/golden/classification_cases.json` (100 test cases)
2. **Evaluate**: For each case, sends the input to the configured LLM with the
   versioned prompt and captures the model's classification output
3. **Score**: Compares model outputs against expected labels to compute accuracy
4. **Compare** (optional): If `--baseline-version` is provided, retrieves the
   latest successful run of that version from SQLite and computes per-case deltas
5. **Metrics**: The MetricsEngine auto-selects appropriate metric calculators
   based on evaluator type (e.g., precision/recall/F1 for classification)
6. **Report**: Generates an HTML report with pass/fail status, metrics table,
   metric deltas, regression check results, and a case-level breakdown
7. **Persist**: Saves the full run (cases, metrics, status) to SQLite
8. **Notify**: Optionally sends a Slack message with the run status
9. **Exit**: Returns `0` (passed) or `1` (failed) based on regression thresholds

## License

MIT