"""Utility functions for reading and writing Zotero libraries."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from env_utils import load_env
from zotero_sync import (
    ZoteroClient,
    get_env_config,
    sync_papers_to_zotero,
)


load_env()

def _client_from_env() -> ZoteroClient:
    config = get_env_config()
    if not config:
        raise EnvironmentError("Zotero environment variables are not configured.")
    return ZoteroClient(config)


def health_check() -> Dict[str, Any]:
    client = _client_from_env()
    collections = client.list_collections()
    return {"status": "ok", "collections": len(collections)}


def list_collections() -> List[Dict[str, Any]]:
    client = _client_from_env()
    return client.list_collections()


def rename_collection(collection_key: str, new_name: str) -> Dict[str, Any]:
    client = _client_from_env()
    client.update_collection_name(collection_key, new_name)
    return {"status": "ok", "collection_key": collection_key, "new_name": new_name}


def rename_collection_by_name(old_name: str, new_name: str) -> Dict[str, Any]:
    client = _client_from_env()
    collection = client.find_collection_by_name(old_name)
    if not collection:
        raise ValueError(f"Collection not found: {old_name}")
    key = collection.get("key")
    client.update_collection_name(key, new_name)
    return {"status": "ok", "collection_key": key, "new_name": new_name}


def list_items(collection_key: Optional[str] = None, limit: int = 50, start: int = 0) -> List[Dict[str, Any]]:
    client = _client_from_env()
    return client.list_items(collection_key=collection_key, limit=limit, start=start)


def search_items(query: str, collection_key: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    client = _client_from_env()
    return client.search_items(query=query, collection_key=collection_key, limit=limit)


def get_item(item_key: str) -> Dict[str, Any]:
    client = _client_from_env()
    return client.get_item(item_key)


def sync_from_metadata(
    metadata_path: Path,
    topic: str,
    download_log_path: Optional[Path] = None,
    pdf_dir: Optional[Path] = None,
    collection_override: Optional[str] = None,
    attach_pdfs: bool = False,
    dedupe_local: bool = True,
    dedupe_remote: bool = True,
) -> Dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    papers = metadata_path.read_text()
    import json

    payload = json.loads(papers)
    if not isinstance(payload, list):
        raise ValueError("Metadata JSON must be a list of papers.")
    return sync_papers_to_zotero(
        payload,
        topic=topic,
        download_log_path=download_log_path,
        pdf_dir=pdf_dir,
        collection_override=collection_override,
        attach_pdfs=attach_pdfs,
        dedupe_local=dedupe_local,
        dedupe_remote=dedupe_remote,
    )
