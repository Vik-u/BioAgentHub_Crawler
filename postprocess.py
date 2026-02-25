"""Post-processing utilities for crawler outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from zotero_sync import _paper_key


def load_papers(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            papers.extend([p for p in data if isinstance(p, dict)])
    return papers


def dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for paper in papers:
        key = _paper_key(paper) or str(hash(json.dumps(paper, sort_keys=True)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique


def merge_and_dedupe(metadata_paths: Iterable[Path], output_path: Path) -> Dict[str, Any]:
    papers = load_papers(metadata_paths)
    deduped = dedupe_papers(papers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(deduped, indent=2))
    return {
        "input_files": [str(p) for p in metadata_paths],
        "output_path": str(output_path),
        "total": len(papers),
        "deduped": len(deduped),
    }
