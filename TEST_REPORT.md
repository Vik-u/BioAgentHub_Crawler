# BioAgentHub_Crawler Test & Hygiene Report

Date: Tue Feb 17 22:40:03 CST 2026

## Test Runs

### Full Suite (unit + CLI + integration)
Command:
```bash
set -a
source /taiga/illinois/eng/chbe/zhao5/vikas/iBF/BioAgentHub_iBF/.env
set +a
RUN_LIVE_INTEGRATION=1 pytest -q
```
Result:
- 9 passed
- 5 warnings
- Duration: ~4 minutes

### Warnings Observed
- CrewAI/Pydantic v2 deprecation warnings (class-based config, v1/v2 mixing). Upstream; does not affect pass/fail.
- Gradio dependency warnings:
  - `python_multipart` import deprecation.
  - `bottleneck` version warning from pandas.

## Capability Coverage

Verified via tests and smoke checks:
- CLI entrypoints: `simple_crawl.py`, `agentic_crawl.py`, `api_server.py`, `learn_weights.py`, `gradio_crawler.py`
- Live crawler integration (network): simple + agentic
- FastAPI app health and simple crawl endpoint tests

## Hygiene & Security Checks

Checks performed:
- TODO/FIXME scan: none found in Python sources.
- Secret scan (repo): no API keys found in tracked files; only `.env.example` contains placeholder values.
- `.gitignore` includes `.env`, output folders, logs, and PDFs.
- Outputs/logs are generated at runtime and ignored by git.

Potential warnings (non-blocking):
- `.venv` exists locally; ensure it remains ignored.
- Vendor wheel `vendor_wheels/crewai-0.5.0-py3-none-any.whl` is used for local installs; packaging uses `requirements-packaging.txt`.

## Current Status

- Tests: PASS (full suite)
- Integration: PASS with live OpenAI key
- Repo hygiene: clean; no sensitive data detected in tracked files

