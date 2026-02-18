import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _require_live():
    if os.getenv("RUN_LIVE_INTEGRATION") != "1":
        pytest.skip("Set RUN_LIVE_INTEGRATION=1 to run live integration tests")


def _run(cmd, timeout=300):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)


@pytest.mark.integration
def test_simple_crawl_live(tmp_path):
    _require_live()
    result = _run(
        [
            sys.executable,
            str(ROOT / "simple_crawl.py"),
            "--query",
            "PETase depolymerase",
            "--max",
            "1",
            "--download",
            "0",
            "--out",
            str(tmp_path),
        ],
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_agentic_crawl_live(tmp_path):
    _require_live()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    config_path = tmp_path / "agentic_config_smoke.yaml"
    config_path.write_text(
        """llm:
  provider: openai
  model: gpt-4o-mini
search:
  target_results: 2
scoring:
  enable_embeddings: false
  enable_cross_encoder: false
""",
        encoding="utf-8",
    )

    result = _run(
        [
            sys.executable,
            str(ROOT / "agentic_crawl.py"),
            "--brief",
            "PETase thermostability",
            "--max-results",
            "2",
            "--downloads",
            "0",
            "--min-queries",
            "1",
            "--max-queries",
            "1",
            "--recall-cap",
            "3",
            "--precision-cap",
            "3",
            "--output",
            str(tmp_path),
            "--config",
            str(config_path),
        ],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
