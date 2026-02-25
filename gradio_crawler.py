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
from zotero_library import list_collections as zotero_list_collections
from zotero_library import search_items as zotero_search_items


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


def _zotero_api_get(api_url: str, path: str, params: Dict[str, Any] | None = None) -> Any:
    if not api_url:
        raise ValueError("API URL is required for remote Zotero calls.")
    url = f"{api_url.rstrip('/')}{path}"
    resp = requests.get(url, params=params or {}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_zotero_list_collections(api_url: str) -> str:
    try:
        if api_url:
            data = _zotero_api_get(api_url, "/zotero/collections")
        else:
            data = zotero_list_collections()
        return json.dumps(data, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


def run_zotero_search(api_url: str, query: str, collection_key: str) -> str:
    try:
        if api_url:
            params = {"query": query}
            if collection_key:
                params["collection_key"] = collection_key
            data = _zotero_api_get(api_url, "/zotero/search", params=params)
        else:
            data = zotero_search_items(query=query, collection_key=collection_key or None)
        return json.dumps(data, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)




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
    with_pdf: bool,
    no_zotero: bool,
    gatekeeper_mode: str,
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
        "with_pdf": with_pdf,
        "no_zotero": no_zotero,
        "gatekeeper_mode": gatekeeper_mode,
    }
    resp = requests.post(api_url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _build_table_html(metadata_path: str | None, limit: int = 50) -> str:
    if not metadata_path:
        return "<p>No metadata available.</p>"
    path = Path(metadata_path)
    if not path.exists():
        return "<p>Metadata file not found.</p>"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return "<p>Failed to read metadata.</p>"

    headers = ["Title", "DOI", "Source", "Published", "Open access", "Score"]
    rows = []
    for paper in (data or [])[:limit]:
        title = paper.get("title") or ""
        doi = paper.get("doi") or ""
        source = paper.get("source") or ""
        published = paper.get("published") or ""
        open_access = "Yes" if paper.get("open_access") else "No"
        score = paper.get("final_score")
        if score is None:
            score = paper.get("relevance_score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else ""

        link = ""
        if doi:
            link = f"https://doi.org/{doi}"
        elif paper.get("pmcid"):
            link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper.get('pmcid')}/"
        elif paper.get("pmid"):
            link = f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/"

        if link:
            title_html = f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>'
        else:
            title_html = title

        rows.append([title_html, doi, source, published, open_access, score_str])

    def _cell(value: str) -> str:
        return f"<td>{value}</td>"

    table_html = ["<table style=\"width:100%; border-collapse: collapse;\">"]
    table_html.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
    table_html.append("<tbody>")
    for row in rows:
        table_html.append("<tr>" + "".join(_cell(val) for val in row) + "</tr>")
    table_html.append("</tbody></table>")
    return "\n".join(table_html)


def _build_funnel_markdown(metadata_path: str | None, download_log_path: str | None) -> str:
    if not metadata_path:
        return "No metadata available."
    path = Path(metadata_path)
    if not path.exists():
        return "Metadata file not found."
    try:
        data = json.loads(path.read_text())
    except Exception:
        return "Failed to read metadata."

    total = len(data)
    has_doi = sum(1 for paper in data if paper.get("doi"))
    open_access = sum(1 for paper in data if paper.get("open_access"))
    has_pmcid = sum(1 for paper in data if paper.get("pmcid"))
    downloaded = None
    if download_log_path:
        log_path = Path(download_log_path)
        if log_path.exists():
            try:
                logs = json.loads(log_path.read_text())
                downloaded = sum(1 for entry in logs if entry.get("status") == "success")
            except Exception:
                downloaded = None

    lines = [
        "| Stage | Count |",
        "|---|---:|",
        f"| Total papers | {total} |",
        f"| Has DOI | {has_doi} |",
        f"| Open access | {open_access} |",
        f"| Has PMCID | {has_pmcid} |",
    ]
    if downloaded is not None:
        lines.append(f"| PDFs downloaded | {downloaded} |")
    else:
        lines.append("| PDFs downloaded | N/A |")
    return "\n".join(lines)


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
    with_pdf: bool,
    no_zotero: bool,
    gatekeeper_mode: str,
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
                with_pdf=with_pdf,
                no_zotero=no_zotero,
                gatekeeper_mode=gatekeeper_mode,
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
            zotero_collection=None,
            no_zotero=no_zotero,
            with_pdf=with_pdf,
            no_zotero_local_dedupe=False,
            no_zotero_remote_dedupe=False,
            gatekeeper=gatekeeper_mode,
        )
        config = load_config(args.config)
        outputs = run_agentic_pipeline(args, config)
        result = {"run_dir": outputs.get("run_dir", ""), "result": outputs}

    result_payload = result.get("result", {})
    search = result_payload.get("search", {}) if isinstance(result_payload, dict) else {}
    downloads = result_payload.get("downloads", {}) if isinstance(result_payload, dict) else {}
    table_html = _build_table_html(search.get("metadata_path"))
    funnel_md = _build_funnel_markdown(search.get("metadata_path"), downloads.get("log_path"))
    summary_lines = [
        f"Total papers: {search.get('total_records', 0)}",
        f"Primary anchor: {search.get('primary_anchor', '-')}",
        f"Metadata path: {search.get('metadata_path', '-')}",
        f"Run dir: {result.get('run_dir', '-')}",
    ]
    return "\n".join(summary_lines), funnel_md, table_html


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
            with_pdf = gr.Checkbox(label="Attach PDFs to Zotero", value=False)
            no_zotero = gr.Checkbox(label="Disable Zotero sync", value=False)
        with gr.Row():
            gatekeeper_mode = gr.Dropdown(
                choices=["skip", "refresh", "off"],
                value="skip",
                label="Zotero gatekeeper",
            )
        with gr.Row():
            api_url = gr.Textbox(
                label="API base URL (optional)",
                value="http://127.0.0.1:8005",
            )

        run_btn = gr.Button("Run crawl", variant="primary")
        summary = gr.Textbox(label="Summary", lines=8)
        funnel_stats = gr.Markdown(label="Funnel stats")
        result_table = gr.HTML()

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
                with_pdf,
                no_zotero,
                gatekeeper_mode,
            ],
            outputs=[summary, funnel_stats, result_table],
        )

        with gr.Accordion("Zotero Tools", open=False):
            zotero_query = gr.Textbox(label="Zotero search query", placeholder="e.g., PETase")
            zotero_collection = gr.Textbox(label="Collection key (optional)")
            zotero_output = gr.Textbox(label="Zotero output", lines=12)
            with gr.Row():
                zotero_list_btn = gr.Button("List collections")
                zotero_search_btn = gr.Button("Search")

            zotero_list_btn.click(
                fn=run_zotero_list_collections,
                inputs=[api_url],
                outputs=[zotero_output],
            )
            zotero_search_btn.click(
                fn=run_zotero_search,
                inputs=[api_url, zotero_query, zotero_collection],
                outputs=[zotero_output],
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
