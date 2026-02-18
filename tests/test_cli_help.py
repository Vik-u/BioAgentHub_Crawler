import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script_name), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_simple_crawl_help():
    result = _run_help("simple_crawl.py")
    assert result.returncode == 0, result.stderr


def test_agentic_crawl_help():
    result = _run_help("agentic_crawl.py")
    assert result.returncode == 0, result.stderr


def test_api_server_help():
    result = _run_help("api_server.py")
    assert result.returncode == 0, result.stderr


def test_learn_weights_help():
    result = _run_help("learn_weights.py")
    assert result.returncode == 0, result.stderr


def test_gradio_help():
    try:
        import gradio  # noqa: F401
    except Exception:
        pytest.skip("gradio not installed")
    result = _run_help("gradio_crawler.py")
    assert result.returncode == 0, result.stderr
