#!/usr/bin/env python3
"""Gradio UI for the simple crawler."""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

import requests

from agentic_crawl import load_config, run_agentic_pipeline


def _make_run_dir(output_root: str, run_name: str | None) -> Path:
    root = Path(output_root)
    stamp = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return root / stamp




def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_api_endpoint(api_url: str, mode: str) -> str:
    if not api_url:
        return ""
    raw = api_url.strip()
    if not raw:
        return ""
    if "/crawl/" in raw:
        return raw
    return f"{raw.rstrip('/')}/crawl/{mode}"




def _run_agentic_via_api(
    api_url: str,
    brief: str,
    max_results: int,
    downloads: int,
    min_queries: int,
    max_queries: int,
    target_papers: int | None,
    require_primary_anchor: bool,
    model: str,
    config_path: str,
    output_root: str,
    run_name: str | None,
) -> Dict[str, Any]:
    payload = {
        "brief": brief,
        "max_results": max_results,
        "downloads": downloads,
        "min_queries": min_queries,
        "max_queries": max_queries,
        "target_papers": target_papers,
        "require_primary_anchor": require_primary_anchor,
        "model": model,
        "config_path": config_path,
        "output_root": output_root,
        "run_name": run_name,
    }
    resp = requests.post(api_url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _build_table_from_metadata(metadata_path: str | None, limit: int = 50) -> List[List[str]]:
    if not metadata_path:
        return []
    path = Path(metadata_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    rows: List[List[str]] = []
    for paper in (data or [])[:limit]:
        doi = paper.get("doi") or ""
        link = ""
        if doi:
            link = f"https://doi.org/{doi}"
        elif paper.get("pmcid"):
            link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper.get('pmcid')}/"
        elif paper.get("pmid"):
            link = f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/"
        authors = paper.get("authors") or []
        if isinstance(authors, list):
            authors = ", ".join(authors[:3])
        rows.append([paper.get("title") or "", doi, link, authors or ""])
    return rows


def run_crawl(
    brief: str,
    max_results: int,
    download_count: int,
    min_queries: int,
    max_queries: int,
    target_papers: str,
    require_primary_anchor: bool,
    model: str,
    config_path: str,
    output_root: str,
    run_name: str,
    api_url: str,
) -> Tuple[str, Dict[str, Any], List[List[str]]]:
    max_results = _coerce_int(max_results, 10)
    download_count = _coerce_int(download_count, 0)
    output_root = output_root or "agentic_outputs"
    run_name = run_name or None
    table_rows: List[List[str]] = []

    brief = (brief or "").strip()
    if not brief:
        return "Agentic brief is required.", {}, []
    min_queries = _coerce_int(min_queries, 4)
    max_queries = _coerce_int(max_queries, 12)
    target_papers_val = _coerce_int(target_papers, None) if target_papers else None
    result = {}
    endpoint = _resolve_api_endpoint(api_url, "agentic")
    if endpoint:
        try:
            result = _run_agentic_via_api(
                api_url=endpoint,
                brief=brief,
                max_results=max_results,
                downloads=download_count,
                min_queries=min_queries,
                max_queries=max_queries,
                target_papers=target_papers_val,
                require_primary_anchor=require_primary_anchor,
                model=(model or "gpt-4o-mini"),
                config_path=config_path or "agentic_config.yaml",
                output_root=output_root,
                run_name=run_name,
            )
        except Exception:
            result = {}

    if not result:
        args = Namespace(
            brief=brief,
            max_results=max_results,
            downloads=download_count,
            min_queries=min_queries,
            max_queries=max_queries,
            target_papers=target_papers_val,
            recall_cap=40,
            precision_cap=15,
            require_primary_anchor=require_primary_anchor,
            model=(model or "gpt-4o-mini"),
            output=output_root,
            config=config_path or "agentic_config.yaml",
            verbose=False,
        )
        config = load_config(args.config)
        outputs = run_agentic_pipeline(args, config)
        result = {"run_dir": outputs.get("run_dir", ""), "result": outputs}

    result_payload = result.get("result", {})
    search = result_payload.get("search", {}) if isinstance(result_payload, dict) else {}
    table_rows = _build_table_from_metadata(search.get("metadata_path"))
    summary_lines = [
        f"Total records: {search.get('total_records', 0)}",
        f"Primary anchor: {search.get('primary_anchor', '-')}",
        f"Metadata path: {search.get('metadata_path', '-')}",
        f"Run dir: {result.get('run_dir', '-')}",
    ]
    return "\n".join(summary_lines), result, table_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI for BioAgentHub crawler.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=7862, help="Port to bind.")
    parser.add_argument("--share", action="store_true", help="Create a public share link.")
    args = parser.parse_args()

    with gr.Blocks(title="BioAgentHub Crawler") as demo:
        gr.Markdown("# BioAgentHub Crawler")
        gr.Markdown("Agentic mode is default. Simple mode runs the lightweight crawler only.")

        with gr.Row():
            brief = gr.Textbox(
                label="Agentic brief",
                placeholder="e.g., PETase enzyme engineering to improve thermostability and activity; mutagenesis, high-throughput assays",
            )
        with gr.Row():
            max_results = gr.Number(label="Max results per source / cap", value=10, precision=0)
            download_count = gr.Number(label="PDFs to download", value=0, precision=0)
        with gr.Row():
            min_queries = gr.Number(label="Min queries (agentic)", value=4, precision=0)
            max_queries = gr.Number(label="Max queries (agentic)", value=12, precision=0)
        with gr.Row():
            target_papers = gr.Textbox(label="Target papers (agentic)", placeholder="e.g., 12")
            require_primary_anchor = gr.Checkbox(label="Require primary anchor (agentic)", value=False)
        with gr.Row():
            model = gr.Textbox(label="Agent model", value="gpt-4o-mini")
            config_path = gr.Textbox(label="Config path", value="agentic_config.yaml")
        with gr.Row():
            output_root = gr.Textbox(label="Output folder", value="agentic_outputs")
            run_name = gr.Textbox(label="Run name (optional)", placeholder="run_custom_name")
        with gr.Row():
            api_url = gr.Textbox(
                label="API base URL (optional)",
                value="http://127.0.0.1:8005",
            )

        run_btn = gr.Button("Run crawl", variant="primary")
        summary = gr.Textbox(label="Summary", lines=8)
        result_json = gr.JSON(label="Result JSON")
        result_table = gr.Dataframe(
            label="Results table",
            headers=["title", "doi", "link", "authors"],
            datatype=["str", "str", "str", "str"],
            row_count=5,
            col_count=(4, "fixed"),
            wrap=True,
        )

        run_btn.click(
            fn=run_crawl,
            inputs=[
                brief,
                max_results,
                download_count,
                min_queries,
                max_queries,
                target_papers,
                require_primary_anchor,
                model,
                config_path,
                output_root,
                run_name,
                api_url,
            ],
            outputs=[summary, result_json, result_table],
        )

    app, local_url, share_url = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True,
    )
    print(f"Local URL: {local_url}")
    if share_url:
        print(f"Public URL: {share_url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
