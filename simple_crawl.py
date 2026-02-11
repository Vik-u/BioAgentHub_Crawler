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
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tools import biorxiv_search_func, europepmc_search_func
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


def run(query: str, max_results: int, anchor: str, download_count: int, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
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

    return {
        "metadata": str(meta_path),
        "downloads": str(log_path),
        "count": len(deduped),
        "stats": {
            "sources": source_stats,
            "source_errors": source_errors,
            "total": len(results),
            "deduped": len(deduped),
            "download_attempted": len(download_logs),
            "download_succeeded": download_succeeded,
            "download_failed": download_failed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal paper crawler (Europe PMC + bioRxiv).")
    parser.add_argument("--query", required=True, help="Search query (enzyme/topic).")
    parser.add_argument("--anchor", default=None, help="Optional anchor term required in title/abstract.")
    parser.add_argument("--max", type=int, default=10, help="Max results per source.")
    parser.add_argument("--download", type=int, default=1, help="How many papers to attempt downloading.")
    parser.add_argument("--out", type=str, default="crawler_outputs", help="Output directory for results.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"run_{timestamp}"
    run(args.query, args.max, args.anchor, args.download, out_dir)


if __name__ == "__main__":
    main()
