import json
from pathlib import Path

from postprocess import merge_and_dedupe


def test_merge_and_dedupe(tmp_path: Path):
    file_a = tmp_path / "a.json"
    file_b = tmp_path / "b.json"
    file_a.write_text(json.dumps([{"title": "A", "doi": "10.1/abc"}]))
    file_b.write_text(json.dumps([{"title": "A dup", "doi": "10.1/abc"}, {"title": "B"}]))

    output = tmp_path / "out.json"
    result = merge_and_dedupe([file_a, file_b], output)
    assert result["total"] == 3
    assert result["deduped"] == 2
    assert output.exists()
