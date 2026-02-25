import json
from pathlib import Path

import zotero_sync


class DummyClient:
    def __init__(self, config):
        self.config = config
        self.created = []

    def list_collections(self):
        return []

    def get_or_create_collection(self, name, parent_key=None):
        return "COLL123"

    def find_existing_item(self, paper, collection_key=None):
        return None

    def create_item(self, item):
        self.created.append(item)
        return f"ITEM{len(self.created)}"

    def attach_pdf(self, item_key, pdf_path):
        return None


def test_sync_dedupe_local(monkeypatch, tmp_path):
    monkeypatch.setenv("ZOTERO_CACHE_DIR", str(tmp_path))
    config = zotero_sync.ZoteroConfig(
        api_key="key",
        library_type="group",
        library_id="123",
        collection_prefix="test",
    )
    monkeypatch.setattr(zotero_sync, "_get_env_config", lambda: config)
    monkeypatch.setattr(zotero_sync, "_get_env_configs", lambda: [config])
    monkeypatch.setattr(zotero_sync, "ZoteroClient", DummyClient)

    papers = [
        {"title": "Paper A", "doi": "10.1/abc"},
        {"title": "Paper A duplicate", "doi": "10.1/abc"},
    ]
    result = zotero_sync.sync_papers_to_zotero(
        papers,
        topic="test",
        attach_pdfs=False,
        dedupe_local=True,
        dedupe_remote=False,
    )
    assert result["items_created"] == 1
    assert result["skipped_local"] == 1

    cache_files = list(Path(tmp_path).glob("*.json"))
    assert cache_files, "Expected cache file to be written"
    cached = json.loads(cache_files[0].read_text())
    assert cached.get("keys")
