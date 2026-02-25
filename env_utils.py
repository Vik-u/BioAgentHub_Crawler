"""Lightweight .env loader for local runs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip().strip("'").strip('"')
    return key, value


def load_env(path: Optional[Path] = None) -> Optional[Path]:
    """Load variables from a .env file if present. Does not override existing env."""
    dotenv_path = path or (Path(__file__).resolve().parent / ".env")
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return None
    try:
        with open(dotenv_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parsed = _parse_env_line(line)
                if not parsed:
                    continue
                key, value = parsed
                existing = os.environ.get(key)
                if existing is None or existing == "":
                    os.environ[key] = value
        return dotenv_path
    except OSError:
        return None
