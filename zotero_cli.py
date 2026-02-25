"""CLI utilities for Zotero library operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from postprocess import merge_and_dedupe
from zotero_library import (
    get_item,
    health_check,
    list_collections,
    list_items,
    rename_collection,
    rename_collection_by_name,
    search_items,
    sync_from_metadata,
)


def _print(data) -> None:
    print(json.dumps(data, indent=2))


def _cmd_health(_args: argparse.Namespace) -> None:
    _print(health_check())


def _cmd_collections(_args: argparse.Namespace) -> None:
    _print(list_collections())


def _cmd_items(args: argparse.Namespace) -> None:
    _print(list_items(collection_key=args.collection, limit=args.limit, start=args.start))


def _cmd_search(args: argparse.Namespace) -> None:
    _print(search_items(query=args.query, collection_key=args.collection, limit=args.limit))


def _cmd_get(args: argparse.Namespace) -> None:
    _print(get_item(args.item_key))


def _cmd_rename(args: argparse.Namespace) -> None:
    if args.collection_key:
        result = rename_collection(args.collection_key, args.new_name)
    else:
        if not args.old_name:
            raise SystemExit("--old-name is required if --collection-key is not provided.")
        result = rename_collection_by_name(args.old_name, args.new_name)
    _print(result)


def _cmd_sync(args: argparse.Namespace) -> None:
    result = sync_from_metadata(
        metadata_path=Path(args.metadata),
        topic=args.topic,
        download_log_path=Path(args.download_log) if args.download_log else None,
        pdf_dir=Path(args.pdf_dir) if args.pdf_dir else None,
        collection_override=args.collection,
        attach_pdfs=args.with_pdf,
        dedupe_local=not args.no_dedupe_local,
        dedupe_remote=not args.no_dedupe_remote,
    )
    _print(result)


def _cmd_dedupe(args: argparse.Namespace) -> None:
    paths: List[Path] = [Path(p) for p in args.metadata]
    result = merge_and_dedupe(paths, Path(args.output))
    _print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zotero utilities for BioAgentHub Crawler.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check Zotero connectivity.").set_defaults(func=_cmd_health)
    subparsers.add_parser("collections", help="List Zotero collections.").set_defaults(func=_cmd_collections)

    items = subparsers.add_parser("items", help="List items from a collection or library.")
    items.add_argument("--collection", default=None, help="Collection key to filter items.")
    items.add_argument("--limit", type=int, default=50)
    items.add_argument("--start", type=int, default=0)
    items.set_defaults(func=_cmd_items)

    search = subparsers.add_parser("search", help="Search items in the library.")
    search.add_argument("--query", required=True, help="Search query.")
    search.add_argument("--collection", default=None, help="Collection key to filter items.")
    search.add_argument("--limit", type=int, default=20)
    search.set_defaults(func=_cmd_search)

    get_item_cmd = subparsers.add_parser("get", help="Get a single item by key.")
    get_item_cmd.add_argument("--item-key", required=True, help="Zotero item key.")
    get_item_cmd.set_defaults(func=_cmd_get)

    sync = subparsers.add_parser("sync", help="Sync metadata JSON into Zotero.")
    sync.add_argument("--metadata", required=True, help="Path to metadata JSON (list of papers).")
    sync.add_argument("--topic", required=True, help="Topic or project name for collection.")
    sync.add_argument("--collection", default=None, help="Override collection name.")
    sync.add_argument("--download-log", default=None, help="Path to download_log.json")
    sync.add_argument("--pdf-dir", default=None, help="Directory with PDFs.")
    sync.add_argument("--with-pdf", action="store_true", help="Attach PDFs when available.")
    sync.add_argument("--no-dedupe-local", action="store_true", help="Disable local dedupe/cache.")
    sync.add_argument("--no-dedupe-remote", action="store_true", help="Disable Zotero remote dedupe.")
    sync.set_defaults(func=_cmd_sync)

    dedupe = subparsers.add_parser("dedupe", help="Merge and dedupe multiple metadata files.")
    dedupe.add_argument("--metadata", nargs="+", required=True, help="Metadata JSON paths.")
    dedupe.add_argument("--output", required=True, help="Output JSON path.")
    dedupe.set_defaults(func=_cmd_dedupe)

    rename = subparsers.add_parser("rename-collection", help="Rename a collection by key or name.")
    rename.add_argument("--collection-key", default=None, help="Collection key to rename.")
    rename.add_argument("--old-name", default=None, help="Existing collection name (if no key).")
    rename.add_argument("--new-name", required=True, help="New collection name.")
    rename.set_defaults(func=_cmd_rename)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
