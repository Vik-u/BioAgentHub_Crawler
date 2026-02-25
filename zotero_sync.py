"""Zotero sync helpers for BioAgentHub crawler outputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


ZOTERO_BASE_URL = "https://api.zotero.org"
DEFAULT_COLLECTION_PREFIX = ""
DEFAULT_CACHE_DIR = ".zotero_cache"
PETASE_ALIASES = ("petase",)


@dataclass
class ZoteroConfig:
    api_key: str
    library_type: str
    library_id: str
    collection_prefix: str
    base_url: str = ZOTERO_BASE_URL


def _clean_topic(raw: str, fallback: str = "Untitled") -> str:
    cleaned = " ".join((raw or "").replace("\n", " ").split()).strip()
    if not cleaned:
        cleaned = fallback
    return cleaned[:120]


def build_collection_name(topic: str, override: Optional[str], prefix: str) -> str:
    if override:
        base = _clean_topic(override)
    else:
        topic_lower = (topic or "").lower()
        if any(alias in topic_lower for alias in PETASE_ALIASES):
            base = "PETase"
        else:
            base = _clean_topic(topic)
    if prefix:
        return f"{prefix}-{base}"
    return base


def _normalize_collection_name(name: str) -> str:
    raw = (name or "").lower().strip()
    for prefix in ("ibiofoundry-ai-", "ibiofoundry-ai_", "ibiofoundry-ai "):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return " ".join(raw.split())


def _topic_tokens(topic: str, override: Optional[str]) -> List[str]:
    if override:
        return _normalize_collection_name(override).split()
    raw = (topic or "").lower()
    if any(alias in raw for alias in PETASE_ALIASES):
        return ["petase"]
    return _normalize_collection_name(topic).split()


def _tokens_in_order(phrase_tokens: List[str], query_tokens: List[str]) -> bool:
    if not phrase_tokens:
        return False
    if len(phrase_tokens) > len(query_tokens):
        return False
    for idx in range(len(query_tokens) - len(phrase_tokens) + 1):
        if query_tokens[idx : idx + len(phrase_tokens)] == phrase_tokens:
            return True
    return False


def _collection_matches_target(name: str, target: str) -> bool:
    norm = _normalize_collection_name(name)
    if not norm:
        return False
    if norm == target:
        return True
    if target == "petase" and "petase" in norm.split():
        return True
    return False


def _collection_matches_query(name: str, query_tokens: List[str]) -> bool:
    norm = _normalize_collection_name(name)
    if not norm:
        return False
    tokens = norm.split()
    if not tokens:
        return False
    if "petase" in query_tokens and "petase" in tokens:
        return True
    return _tokens_in_order(tokens, query_tokens)


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return " ".join(cleaned.split())


def _paper_key(paper: Dict[str, Any]) -> Optional[str]:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pmid = (paper.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"
    title = _normalize_title(paper.get("title") or "")
    if title:
        return f"title:{title}"
    return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")


def _cache_dir() -> Path:
    raw = os.getenv("ZOTERO_CACHE_DIR")
    return Path(raw).expanduser() if raw else Path(DEFAULT_CACHE_DIR)


def _cache_path(collection_key: str) -> Path:
    cache_root = _cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{_slug(collection_key)}.json"


def _load_cache(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except Exception:
        return set()
    if isinstance(data, list):
        return {str(item) for item in data}
    if isinstance(data, dict):
        return {str(item) for item in data.get("keys", [])}
    return set()


def _save_cache(path: Path, keys: Set[str]) -> None:
    payload = {"keys": sorted(keys)}
    path.write_text(json.dumps(payload, indent=2))


def _split_author(name: str) -> Dict[str, str]:
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        if first:
            return {"creatorType": "author", "firstName": first, "lastName": last}
        return {"creatorType": "author", "lastName": last}
    parts = [p for p in name.strip().split() if p]
    if len(parts) == 1:
        return {"creatorType": "author", "lastName": parts[0]}
    return {"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]}


def _coerce_authors(value: Any) -> List[Dict[str, str]]:
    if not value:
        return []
    if isinstance(value, str):
        return [_split_author(value)]
    if isinstance(value, list):
        authors: List[Dict[str, str]] = []
        for entry in value:
            if not entry:
                continue
            if isinstance(entry, str):
                authors.append(_split_author(entry))
            elif isinstance(entry, dict):
                last = entry.get("lastName") or entry.get("last") or entry.get("family")
                first = entry.get("firstName") or entry.get("first") or entry.get("given")
                if last or first:
                    creator = {"creatorType": "author"}
                    if first:
                        creator["firstName"] = str(first)
                    if last:
                        creator["lastName"] = str(last)
                    authors.append(creator)
        return authors
    return []


def _paper_url(paper: Dict[str, Any]) -> str:
    doi = (paper.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    pmcid = (paper.get("pmcid") or "").strip()
    if pmcid:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    pmid = (paper.get("pmid") or "").strip()
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return ""


def _build_extra(paper: Dict[str, Any]) -> str:
    lines = []
    for key in ("pmid", "pmcid"):
        value = (paper.get(key) or "").strip()
        if value:
            lines.append(f"{key.upper()}: {value}")
    source = (paper.get("source") or "").strip()
    if source:
        lines.append(f"Source: {source}")
    return "\n".join(lines)


def _get_env_config() -> Optional[ZoteroConfig]:
    configs = _get_env_configs()
    return configs[0] if configs else None


def _get_env_configs() -> List[ZoteroConfig]:
    api_key = os.getenv("ZOTERO_API_KEY")
    if not api_key:
        return []
    library_type = (os.getenv("ZOTERO_LIBRARY_TYPE") or "group").strip().lower()
    raw_prefix = os.getenv("ZOTERO_COLLECTION_PREFIX")
    collection_prefix = raw_prefix.strip() if raw_prefix is not None else DEFAULT_COLLECTION_PREFIX
    configs: List[ZoteroConfig] = []
    if library_type == "group":
        primary_group = os.getenv("ZOTERO_GROUP_ID")
        if primary_group:
            configs.append(
                ZoteroConfig(
                    api_key=api_key,
                    library_type=library_type,
                    library_id=primary_group,
                    collection_prefix=collection_prefix,
                    base_url=os.getenv("ZOTERO_BASE_URL") or ZOTERO_BASE_URL,
                )
            )
        secondary_group = os.getenv("ZOTERO_GROUP_ID_SECONDARY")
        if secondary_group:
            configs.append(
                ZoteroConfig(
                    api_key=api_key,
                    library_type=library_type,
                    library_id=secondary_group,
                    collection_prefix=collection_prefix,
                    base_url=os.getenv("ZOTERO_BASE_URL") or ZOTERO_BASE_URL,
                )
            )
    else:
        user_id = os.getenv("ZOTERO_USER_ID")
        if user_id:
            configs.append(
                ZoteroConfig(
                    api_key=api_key,
                    library_type=library_type,
                    library_id=user_id,
                    collection_prefix=collection_prefix,
                    base_url=os.getenv("ZOTERO_BASE_URL") or ZOTERO_BASE_URL,
                )
            )
    if not configs:
        return []
    return configs


def get_env_config() -> Optional[ZoteroConfig]:
    return _get_env_config()


class ZoteroClient:
    def __init__(self, config: ZoteroConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Zotero-API-Key": self.config.api_key,
                "Content-Type": "application/json",
                "Zotero-API-Version": "3",
            }
        )

    @property
    def library_root(self) -> str:
        if self.config.library_type == "group":
            return f"{self.config.base_url}/groups/{self.config.library_id}"
        return f"{self.config.base_url}/users/{self.config.library_id}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = path if path.startswith("http") else f"{self.library_root}{path}"
        for attempt in range(3):
            resp = self.session.request(method, url, timeout=60, **kwargs)
            if resp.status_code != 429:
                return resp
            retry_after = resp.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 2
            time.sleep(delay)
        return resp

    def list_collections(self) -> List[Dict[str, Any]]:
        collections: List[Dict[str, Any]] = []
        start = 0
        while True:
            resp = self._request(
                "GET",
                "/collections",
                params={"limit": 100, "start": start},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            collections.extend(batch)
            if len(batch) < 100:
                break
            start += 100
        return collections

    def get_collection(self, key: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/collections/{key}")
        resp.raise_for_status()
        return resp.json()

    def find_collection_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for entry in self.list_collections():
            data = entry.get("data") or {}
            if data.get("name") == name:
                return entry
        return None

    def update_collection_name(self, key: str, new_name: str) -> None:
        current = self.get_collection(key)
        data = current.get("data") or {}
        payload = {
            "name": new_name,
            "parentCollection": data.get("parentCollection") or False,
            "relations": data.get("relations") or {},
        }
        version = current.get("version")
        headers = {}
        if version is not None:
            headers["If-Unmodified-Since-Version"] = str(version)
        resp = self._request("PUT", f"/collections/{key}", json=payload, headers=headers)
        resp.raise_for_status()

    def get_or_create_collection(self, name: str, parent_key: Optional[str] = None) -> str:
        existing = None
        for entry in self.list_collections():
            data = entry.get("data") or {}
            if data.get("name") == name and data.get("parentCollection") == parent_key:
                existing = entry.get("key")
                break
        if existing:
            return existing

        payload: List[Dict[str, Any]] = [{"name": name}]
        if parent_key:
            payload[0]["parentCollection"] = parent_key
        resp = self._request("POST", "/collections", json=payload)
        resp.raise_for_status()
        data = resp.json()
        key = data.get("success", {}).get("0")
        if not key:
            raise RuntimeError("Failed to create Zotero collection.")
        return key

    def create_item(self, item: Dict[str, Any]) -> str:
        resp = self._request("POST", "/items", json=[item])
        resp.raise_for_status()
        data = resp.json()
        key = data.get("success", {}).get("0")
        if not key:
            raise RuntimeError("Failed to create Zotero item.")
        return key

    def list_items(
        self,
        collection_key: Optional[str] = None,
        limit: int = 50,
        start: int = 0,
    ) -> List[Dict[str, Any]]:
        path = f"/collections/{collection_key}/items" if collection_key else "/items"
        resp = self._request("GET", path, params={"limit": limit, "start": start})
        resp.raise_for_status()
        return resp.json()

    def search_items(
        self,
        query: str,
        collection_key: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        path = f"/collections/{collection_key}/items" if collection_key else "/items"
        resp = self._request(
            "GET",
            path,
            params={"limit": limit, "q": query, "qmode": "everything"},
        )
        resp.raise_for_status()
        return resp.json()

    def get_item(self, item_key: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/items/{item_key}")
        resp.raise_for_status()
        return resp.json()

    def find_existing_item(
        self,
        paper: Dict[str, Any],
        collection_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        doi = (paper.get("doi") or "").strip().lower()
        pmid = (paper.get("pmid") or "").strip()
        title = _normalize_title(paper.get("title") or "")
        candidates: List[Dict[str, Any]] = []
        if doi:
            candidates = self.search_items(doi, collection_key=collection_key, limit=20)
            for item in candidates:
                data = item.get("data") or {}
                if (data.get("DOI") or "").strip().lower() == doi:
                    return item
        if pmid:
            candidates = self.search_items(pmid, collection_key=collection_key, limit=20)
            for item in candidates:
                data = item.get("data") or {}
                extra = (data.get("extra") or "")
                if f"PMID: {pmid}" in extra:
                    return item
        if title:
            candidates = self.search_items(title, collection_key=collection_key, limit=10)
            for item in candidates:
                data = item.get("data") or {}
                if _normalize_title(data.get("title") or "") == title:
                    return item
        return None

    def _file_md5(self, path: Path) -> str:
        md5 = hashlib.md5()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def attach_pdf(self, item_key: str, pdf_path: Path) -> Optional[str]:
        if not pdf_path.exists():
            return None
        filename = pdf_path.name
        filesize = pdf_path.stat().st_size
        mtime = int(pdf_path.stat().st_mtime * 1000)
        md5 = self._file_md5(pdf_path)

        attachment_item = {
            "itemType": "attachment",
            "parentItem": item_key,
            "linkMode": "imported_file",
            "title": filename,
            "contentType": "application/pdf",
            "filename": filename,
            "md5": md5,
            "mtime": mtime,
        }
        resp = self._request("POST", "/items", json=[attachment_item])
        resp.raise_for_status()
        data = resp.json()
        attachment_key = data.get("success", {}).get("0")
        if not attachment_key:
            raise RuntimeError("Failed to create Zotero attachment item.")

        auth_payload = {
            "md5": md5,
            "filename": filename,
            "filesize": filesize,
            "mtime": mtime,
            "contentType": "application/pdf",
        }
        auth_resp = self._request(
            "POST",
            f"/items/{attachment_key}/file",
            params={"params": 1},
            json=auth_payload,
            headers={"If-None-Match": "*"},
        )
        auth_resp.raise_for_status()
        auth_data = auth_resp.json()
        if auth_data.get("exists"):
            return attachment_key

        upload_url = auth_data.get("url")
        upload_key = auth_data.get("uploadKey")
        params = auth_data.get("params") or []
        if not upload_url or not upload_key:
            raise RuntimeError("Missing upload authorization for Zotero file.")

        form = {p["name"]: p["value"] for p in params if "name" in p and "value" in p}
        with open(pdf_path, "rb") as handle:
            files = {"file": (filename, handle, "application/pdf")}
            upload_resp = requests.post(upload_url, data=form, files=files, timeout=120)
        upload_resp.raise_for_status()

        register_resp = self._request(
            "POST",
            f"/items/{attachment_key}/file",
            params={"upload": upload_key},
            headers={"If-None-Match": "*"},
        )
        register_resp.raise_for_status()
        return attachment_key


def _merge_collection_items(
    client: ZoteroClient,
    source_key: str,
    target_key: str,
    delay_sec: float = 0.05,
) -> int:
    moved = 0
    start = 0
    while True:
        batch = client.list_items(collection_key=source_key, limit=100, start=start)
        if not batch:
            break
        for item in batch:
            data = item.get("data") or {}
            item_type = data.get("itemType")
            if item_type in ("attachment", "note"):
                continue
            item_key = item.get("key")
            for _ in range(3):
                current = client.get_item(item_key)
                current_data = current.get("data") or {}
                collections = current_data.get("collections") or []
                if target_key in collections:
                    break
                collections.append(target_key)
                current_data["collections"] = collections
                version = current.get("version")
                headers = {"If-Unmodified-Since-Version": str(version)} if version is not None else {}
                resp = client._request("PUT", f"/items/{item_key}", json=current_data, headers=headers)
                if resp.status_code == 409:
                    time.sleep(0.2)
                    continue
                resp.raise_for_status()
                moved += 1
                time.sleep(delay_sec)
                break
        if len(batch) < 100:
            break
        start += 100
    return moved


def _resolve_collection(
    client: ZoteroClient,
    topic: str,
    override: Optional[str],
    prefix: str,
    consolidate: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    target_name = build_collection_name(topic, override, prefix)
    target_norm = _normalize_collection_name(target_name)
    query_tokens = _topic_tokens(topic, override)
    query_norm = _normalize_collection_name(override or topic)
    matches: List[Dict[str, Any]] = []
    for entry in client.list_collections():
        name = (entry.get("data") or {}).get("name") or ""
        if _collection_matches_target(name, target_norm):
            matches.append(entry)
        elif consolidate and query_norm and _collection_matches_query(name, query_tokens):
            matches.append(entry)

    if matches:
        # Prefer exact target name, else highest item count
        exact = [m for m in matches if (m.get("data") or {}).get("name") == target_name]
        if exact:
            canonical = exact[0]
        else:
            canonical = max(matches, key=lambda m: (m.get("meta") or {}).get("numItems", 0))
            if (canonical.get("data") or {}).get("name") != target_name:
                client.update_collection_name(canonical.get("key"), target_name)
        canonical_key = canonical.get("key")
        canonical_items = (canonical.get("meta") or {}).get("numItems", 0)
        merged = []
        if consolidate:
            for col in matches:
                col_key = col.get("key")
                if col_key == canonical_key:
                    continue
                moved = _merge_collection_items(client, col_key, canonical_key)
                col_version = col.get("version")
                headers = {"If-Unmodified-Since-Version": str(col_version)} if col_version is not None else {}
                resp = client._request("DELETE", f"/collections/{col_key}", headers=headers)
                resp.raise_for_status()
                merged.append({"from": (col.get("data") or {}).get("name"), "moved": moved})
        return canonical_key, target_name, {"merged": merged, "num_items": canonical_items}

    key = client.get_or_create_collection(target_name)
    return key, target_name, {"merged": [], "num_items": 0}


def build_item_payload(
    paper: Dict[str, Any],
    collection_key: str,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    published = (paper.get("published") or "").strip()
    doi = (paper.get("doi") or "").strip()
    journal = (paper.get("journal") or "").strip()
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    creators = _coerce_authors(paper.get("authors"))

    payload = {
        "itemType": "journalArticle",
        "title": title,
        "abstractNote": abstract,
        "date": published,
        "DOI": doi,
        "publicationTitle": journal,
        "url": _paper_url(paper),
        "creators": creators,
        "collections": [collection_key],
        "tags": [{"tag": t} for t in (tags or []) if t],
    }

    extra = _build_extra(paper)
    if extra:
        payload["extra"] = extra
    return payload


def sync_papers_to_zotero(
    papers: List[Dict[str, Any]],
    topic: str,
    download_log_path: Optional[Path] = None,
    pdf_dir: Optional[Path] = None,
    collection_override: Optional[str] = None,
    attach_pdfs: bool = False,
    dedupe_local: bool = True,
    dedupe_remote: bool = True,
) -> Dict[str, Any]:
    configs = _get_env_configs()
    if not configs:
        return {"status": "skipped", "reason": "missing_zotero_env"}

    results = []
    for config in configs:
        results.append(
            _sync_papers_to_zotero_single(
                config=config,
                papers=papers,
                topic=topic,
                download_log_path=download_log_path,
                pdf_dir=pdf_dir,
                collection_override=collection_override,
                attach_pdfs=attach_pdfs,
                dedupe_local=dedupe_local,
                dedupe_remote=dedupe_remote,
            )
        )
    if len(results) == 1:
        return results[0]
    return {"status": "ok", "results": results}


def _sync_papers_to_zotero_single(
    config: ZoteroConfig,
    papers: List[Dict[str, Any]],
    topic: str,
    download_log_path: Optional[Path],
    pdf_dir: Optional[Path],
    collection_override: Optional[str],
    attach_pdfs: bool,
    dedupe_local: bool,
    dedupe_remote: bool,
) -> Dict[str, Any]:
    client = ZoteroClient(config)
    collection_key, collection_name, merge_info = _resolve_collection(
        client,
        topic=topic,
        override=collection_override,
        prefix=config.collection_prefix,
        consolidate=True,
    )

    cache_path = _cache_path(collection_key)
    cached_keys = _load_cache(cache_path) if dedupe_local else set()

    item_keys: Dict[int, str] = {}
    failures: List[str] = []
    skipped_local = 0
    skipped_remote = 0
    to_upload: List[Tuple[int, Dict[str, Any], Optional[str]]] = []
    for idx, paper in enumerate(papers, start=1):
        key = _paper_key(paper)
        if dedupe_local and key and key in cached_keys:
            skipped_local += 1
            continue
        if dedupe_remote:
            try:
                existing = client.find_existing_item(paper, collection_key=collection_key)
            except Exception as exc:
                failures.append(f"dedupe#{idx}:{exc}")
                existing = None
            if existing:
                skipped_remote += 1
                if key:
                    cached_keys.add(key)
                continue
        if dedupe_local and key:
            cached_keys.add(key)
        to_upload.append((idx, paper, key))

    for idx, paper, key in to_upload:
        try:
            tags = []
            source = (paper.get("source") or "").strip()
            if source:
                tags.append(f"source:{source}")
            if topic:
                tags.append(f"topic:{_clean_topic(topic)}")
            item_key = client.create_item(build_item_payload(paper, collection_key, tags=tags))
            item_keys[idx] = item_key
            if key:
                cached_keys.add(key)
        except Exception as exc:
            failures.append(f"item#{idx}:{exc}")

    attachments = []
    if attach_pdfs and download_log_path and download_log_path.exists():
        try:
            logs = json.loads(download_log_path.read_text())
        except Exception:
            logs = []
        if isinstance(logs, dict):
            logs = logs.get("details") or []
        for entry in logs:
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "success":
                continue
            index = entry.get("index")
            filename = entry.get("filename")
            if not index or not filename:
                continue
            item_key = item_keys.get(int(index))
            if not item_key:
                continue
            target_dir = pdf_dir or download_log_path.parent
            pdf_path = Path(target_dir) / filename
            try:
                attachment_key = client.attach_pdf(item_key, pdf_path)
                attachments.append(
                    {
                        "index": index,
                        "item_key": item_key,
                        "attachment_key": attachment_key,
                        "filename": filename,
                    }
                )
            except Exception as exc:
                failures.append(f"attach#{index}:{exc}")

    if dedupe_local:
        _save_cache(cache_path, cached_keys)

    return {
        "status": "ok",
        "library_id": config.library_id,
        "collection": collection_name,
        "collection_key": collection_key,
        "merged": merge_info.get("merged", []),
        "items_created": len(item_keys),
        "skipped_local": skipped_local,
        "skipped_remote": skipped_remote,
        "attachments": len(attachments),
        "failures": failures,
    }


def zotero_configured() -> bool:
    return _get_env_config() is not None


def get_collection_info(topic: str, override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    config = _get_env_config()
    if not config:
        return None
    client = ZoteroClient(config)
    target_name = build_collection_name(topic, override, config.collection_prefix)
    target_norm = _normalize_collection_name(target_name)
    query_tokens = _topic_tokens(topic, override)
    query_norm = _normalize_collection_name(override or topic)
    target_matches: List[Dict[str, Any]] = []
    query_matches: List[Dict[str, Any]] = []
    for entry in client.list_collections():
        name = (entry.get("data") or {}).get("name") or ""
        if _collection_matches_target(name, target_norm):
            target_matches.append(entry)
        elif query_norm and _collection_matches_query(name, query_tokens):
            query_matches.append(entry)
    matches = target_matches or query_matches
    if not matches:
        return {
            "exists": False,
            "name": target_name,
            "key": None,
            "num_items": 0,
        }
    canonical = max(matches, key=lambda m: (m.get("meta") or {}).get("numItems", 0))
    match_reason = "target" if canonical in target_matches else "query"
    return {
        "exists": True,
        "name": (canonical.get("data") or {}).get("name"),
        "key": canonical.get("key"),
        "num_items": (canonical.get("meta") or {}).get("numItems", 0),
        "match_reason": match_reason,
        "matched_collections": [(m.get("data") or {}).get("name") for m in matches],
    }
