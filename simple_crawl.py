#!/usr/bin/env python3
"""
Lightweight literature crawler (subset of PEproject pipeline).
- Searches Europe PMC + bioRxiv with optional anchor term.
- Deduplicates by DOI/PMID/title.
- Attempts PDF download using multi-strategy downloader.

Example:
    python crawler/simple_crawl.py --query "PETase depolymerase" --max 5 \
        --download 1 --out crawler_outputs
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tools import biorxiv_search_func, europepmc_search_func
from env_utils import load_env
from zotero_sync import get_collection_info, sync_papers_to_zotero, zotero_configured


load_env()
from pdf_downloader import ImprovedPDFDownloader


def deduplicate(papers: List[Dict]) -> List[Dict]:
    """Deduplicate by DOI, PMID, or title prefix to avoid redundant downloads."""
    seen_doi = set()
    seen_pmid = set()
    seen_title = set()
    unique = []
    for p in papers:
        doi = (p.get("doi") or "").lower().strip()
        pmid = (p.get("pmid") or "").strip()
        title = (p.get("title") or "").strip().lower()
        title_key = title[:80]
        if doi and doi in seen_doi:
            continue
        if pmid and pmid in seen_pmid:
            continue
        if title_key and title_key in seen_title:
            continue
        if doi:
            seen_doi.add(doi)
        if pmid:
            seen_pmid.add(pmid)
        if title_key:
            seen_title.add(title_key)
        unique.append(p)
    return unique


def run(
    query: str,
    max_results: int,
    anchor: str,
    download_count: int,
    out_dir: Path,
    zotero_collection: Optional[str] = None,
    disable_zotero: bool = False,
    attach_pdfs: bool = False,
    dedupe_local: bool = True,
    dedupe_remote: bool = True,
    gatekeeper_mode: str = "skip",
) -> Dict:
    start_time = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    gatekeeper = None
    if zotero_configured() and gatekeeper_mode != "off":
        gatekeeper = get_collection_info(query, zotero_collection)
        if gatekeeper and gatekeeper.get("exists") and gatekeeper_mode == "skip" and gatekeeper.get("num_items", 0) > 0:
            runtime_sec = round(time.perf_counter() - start_time, 2)
            return {
                "metadata": None,
                "downloads": None,
                "count": 0,
                "skipped": True,
                "reason": "collection_exists",
                "zotero": {"gatekeeper": gatekeeper},
                "stats": {"runtime_sec": runtime_sec},
            }
    print(f"🔎 Searching Europe PMC and bioRxiv for: {query}")
    anchor_arg = anchor if anchor else None

    source_stats = {"Europe PMC": 0, "bioRxiv": 0}
    source_errors = {}

    def collect(source: str, papers: List[Dict]) -> List[Dict]:
        if not papers:
            print(f"   {source}: 0 hits")
            source_stats[source] = 0
            return []
        if isinstance(papers, list) and papers and isinstance(papers[0], dict) and "error" in papers[0]:
            error_msg = papers[0].get("error", "unknown")
            print(f"   {source} error: {error_msg}")
            source_stats[source] = 0
            source_errors[source] = error_msg
            return []
        print(f"   {source}: {len(papers)} hits")
        source_stats[source] = len(papers)
        return papers

    results = []
    results.extend(collect("Europe PMC", europepmc_search_func(query, max_results=max_results, anchor=anchor_arg)))
    results.extend(collect("bioRxiv", biorxiv_search_func(query, max_results=max_results, anchor=anchor_arg)))

    deduped = deduplicate(results)
    print(f"✓ Unique papers: {len(deduped)} (from {len(results)} total)")

    meta_path = out_dir / "papers.json"
    with open(meta_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"📄 Saved metadata to {meta_path}")

    downloader = ImprovedPDFDownloader()
    download_logs = []
    for idx, paper in enumerate(deduped[:download_count], start=1):
        print(f"   ↓ Downloading #{idx}: {paper.get('title','')[:80]}")
        res = downloader.download_paper(paper, out_dir, idx)
        download_logs.append(res)
        print(f"      status={res.get('status')} method={res.get('method','-')} file={res.get('filename','-')}")

    log_path = out_dir / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(download_logs, f, indent=2)
    print(f"🧾 Saved download log to {log_path}")

    download_succeeded = sum(1 for log in download_logs if (log or {}).get("status") == "success")
    download_failed = len(download_logs) - download_succeeded

    zotero_result = None
    if zotero_configured() and not disable_zotero:
        zotero_result = sync_papers_to_zotero(
            deduped,
            topic=query,
            download_log_path=log_path,
            pdf_dir=out_dir,
            collection_override=zotero_collection,
            attach_pdfs=attach_pdfs,
            dedupe_local=dedupe_local,
            dedupe_remote=dedupe_remote,
        )

    runtime_sec = round(time.perf_counter() - start_time, 2)
    return {
        "metadata": str(meta_path),
        "downloads": str(log_path),
        "count": len(deduped),
        "zotero": zotero_result,
        "stats": {
            "sources": source_stats,
            "source_errors": source_errors,
            "total": len(results),
            "deduped": len(deduped),
            "download_attempted": len(download_logs),
            "download_succeeded": download_succeeded,
            "download_failed": download_failed,
            "runtime_sec": runtime_sec,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal paper crawler (Europe PMC + bioRxiv).")
    parser.add_argument("--query", required=True, help="Search query (enzyme/topic).")
    parser.add_argument("--anchor", default=None, help="Optional anchor term required in title/abstract.")
    parser.add_argument("--max", type=int, default=10, help="Max results per source.")
    parser.add_argument("--download", type=int, default=1, help="How many papers to attempt downloading.")
    parser.add_argument("--out", type=str, default="crawler_outputs", help="Output directory for results.")
    parser.add_argument("--zotero-collection", type=str, default=None, help="Override Zotero collection name.")
    parser.add_argument("--no-zotero", action="store_true", help="Disable Zotero sync for this run.")
    parser.add_argument("--with-pdf", action="store_true", help="Attach PDFs to Zotero when available.")
    parser.add_argument("--no-zotero-local-dedupe", action="store_true", help="Disable local dedupe/cache.")
    parser.add_argument("--no-zotero-remote-dedupe", action="store_true", help="Disable Zotero remote dedupe.")
    parser.add_argument(
        "--gatekeeper",
        choices=["skip", "refresh", "off"],
        default="skip",
        help="Zotero gatekeeper mode: skip if collection exists, refresh (crawl anyway), or off.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"run_{timestamp}"
    run(
        args.query,
        args.max,
        args.anchor,
        args.download,
        out_dir,
        zotero_collection=args.zotero_collection,
        disable_zotero=args.no_zotero,
        attach_pdfs=args.with_pdf,
        dedupe_local=not args.no_zotero_local_dedupe,
        dedupe_remote=not args.no_zotero_remote_dedupe,
        gatekeeper_mode=args.gatekeeper,
    )


if __name__ == "__main__":
    main()
