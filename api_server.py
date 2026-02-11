#!/usr/bin/env python3
"""FastAPI wrapper for BioAgentHub_Crawler."""
from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agentic_crawl import load_config, run_agentic_pipeline
from simple_crawl import run as run_simple

app = FastAPI(title="BioAgentHub Crawler API", version="0.1.0")


class SimpleCrawlRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string.")
    max_results: int = Field(10, ge=1, le=500, description="Max results per source.")
    anchor: Optional[str] = Field(None, description="Optional anchor term required in title/abstract.")
    download_count: int = Field(1, ge=0, le=500, description="How many PDFs to attempt downloading.")
    output_root: str = Field("crawler_outputs", description="Base output directory.")
    run_name: Optional[str] = Field(None, description="Optional run folder name; defaults to timestamp.")


class AgenticCrawlRequest(BaseModel):
    brief: str = Field(..., min_length=1, description="Research brief for agentic crawler.")
    max_results: int = Field(8, ge=1, le=500, description="Max records to keep after dedupe.")
    downloads: int = Field(2, ge=0, le=500, description="How many PDFs to attempt.")
    min_queries: int = Field(4, ge=1, le=50, description="Minimum query variants.")
    max_queries: int = Field(12, ge=1, le=50, description="Maximum query variants.")
    target_papers: Optional[int] = Field(None, description="Preferred minimum number of unique papers.")
    recall_cap: int = Field(40, ge=1, le=500, description="Per-query cap for recall queries.")
    precision_cap: int = Field(15, ge=1, le=500, description="Per-query cap for precision queries.")
    require_primary_anchor: bool = Field(False, description="Force primary anchor as hard filter.")
    model: str = Field("gpt-4o-mini", description="OpenAI chat model for agents.")
    output_root: str = Field("agentic_outputs", description="Base output directory.")
    config_path: str = Field("agentic_config.yaml", description="Agentic config file path.")
    verbose: bool = Field(False, description="Enable verbose CrewAI logs.")


class TablePreview(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class SimpleCrawlResponse(BaseModel):
    run_dir: str
    result: dict
    table: Optional[TablePreview] = None


class AgenticCrawlResponse(BaseModel):
    run_dir: str
    result: dict
    table: Optional[TablePreview] = None


def _make_run_dir(output_root: str, run_name: Optional[str]) -> Path:
    root = Path(output_root)
    stamp = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return root / stamp


def _paper_link(paper: Dict[str, Any]) -> str:
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


def _normalize_authors(authors: Any, limit: int = 3) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        trimmed = [str(a) for a in authors if a][:limit]
        return ", ".join(trimmed)
    return str(authors)


def _load_metadata(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    meta_path = Path(path)
    if not meta_path.exists():
        return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _build_table(papers: List[Dict[str, Any]], limit: int = 50) -> TablePreview:
    columns = ["title", "doi", "link", "authors"]
    rows: List[Dict[str, Any]] = []
    for paper in papers[:limit]:
        rows.append(
            {
                "title": paper.get("title") or "",
                "doi": paper.get("doi") or "",
                "link": _paper_link(paper),
                "authors": _normalize_authors(paper.get("authors")),
            }
        )
    return TablePreview(columns=columns, rows=rows)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BioAgentHub Crawler UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #0f172a;
      --panel: #111827;
      --accent: #38bdf8;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --border: #1f2937;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: radial-gradient(circle at top, #1e293b 0%, #0f172a 60%);
      color: var(--text);
    }
    header {
      padding: 28px 24px 8px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0.3px;
    }
    p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 820px;
    }
    main {
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(280px, 380px) 1fr;
      padding: 16px 24px 32px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
    .card {
      background: rgba(17, 24, 39, 0.85);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    label {
      display: block;
      font-size: 12px;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      color: var(--muted);
      margin: 12px 0 6px;
    }
    input, textarea, select {
      width: 100%;
      background: #0b1220;
      border: 1px solid #1f2937;
      color: var(--text);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
    }
    textarea { min-height: 72px; resize: vertical; }
    button {
      margin-top: 16px;
      width: 100%;
      border: none;
      padding: 12px 16px;
      border-radius: 8px;
      background: var(--accent);
      color: #0b1220;
      font-weight: 700;
      letter-spacing: 0.4px;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0 0 8px;
      font-size: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(148, 163, 184, 0.2);
      color: var(--muted);
    }
    pre {
      margin: 0;
      padding: 12px;
      background: #0b1220;
      border-radius: 8px;
      border: 1px solid var(--border);
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.4;
    }
    .result {
      display: grid;
      gap: 12px;
    }
    .ok { color: var(--success); }
    .err { color: var(--danger); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border: 1px solid var(--border);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: rgba(30, 41, 59, 0.7);
    }
  </style>
</head>
<body>
  <header>
    <h1>BioAgentHub Crawler UI</h1>
    <p>Run the simple crawler and inspect results in real time. This UI calls <code>/crawl/simple</code> on the same server.</p>
    <p style="margin-top: 10px;">
      API docs:
      <a href="/docs" style="color: var(--accent);">Swagger UI</a>
      &nbsp;|&nbsp;
      <a href="/redoc" style="color: var(--accent);">ReDoc</a>
    </p>
  </header>
  <main>
    <section class="card">
      <label for="brief">Agentic brief</label>
      <textarea id="brief" placeholder="e.g., PETase enzyme engineering to improve thermostability and activity; mutagenesis, high-throughput assays"></textarea>
      <label for="max_results">Max results per source</label>
      <input id="max_results" type="number" min="1" max="500" value="10" />
      <label for="min_queries">Min queries</label>
      <input id="min_queries" type="number" min="1" max="50" value="4" />
      <label for="max_queries">Max queries</label>
      <input id="max_queries" type="number" min="1" max="50" value="12" />
      <label for="target_papers">Target papers (optional)</label>
      <input id="target_papers" type="number" min="1" max="500" placeholder="e.g., 12" />
      <label for="require_primary_anchor">Require primary anchor</label>
      <select id="require_primary_anchor">
        <option value="false" selected>false</option>
        <option value="true">true</option>
      </select>
      <label for="model">Agent model</label>
      <input id="model" type="text" value="gpt-4o-mini" />
      <label for="config_path">Config path</label>
      <input id="config_path" type="text" value="agentic_config.yaml" />
      <label for="download_count">PDFs to download</label>
      <input id="download_count" type="number" min="0" max="500" value="0" />
      <label for="output_root">Output folder</label>
      <input id="output_root" type="text" value="agentic_outputs" />
      <label for="run_name">Run name (optional)</label>
      <input id="run_name" type="text" placeholder="run_custom_name" />
      <button id="run">Run crawl</button>
    </section>
    <section class="card result">
      <div class="status">
        <span class="pill" id="status-pill">idle</span>
        <span id="status-text">Waiting for a request.</span>
      </div>
      <div id="table"></div>
      <div id="summary"></div>
      <pre id="output">{}</pre>
    </section>
  </main>
  <script>
    const runButton = document.getElementById("run");
    const statusPill = document.getElementById("status-pill");
    const statusText = document.getElementById("status-text");
    const output = document.getElementById("output");
    const summary = document.getElementById("summary");
    const table = document.getElementById("table");

    function setStatus(state, text) {
      statusPill.textContent = state;
      statusPill.className = "pill " + (state === "ok" ? "ok" : state === "error" ? "err" : "");
      statusText.textContent = text;
    }

    function renderTable(tablePayload) {
      if (!tablePayload || !tablePayload.rows || !tablePayload.rows.length) {
        table.innerHTML = "";
        return;
      }
      const cols = tablePayload.columns || ["title", "doi", "link", "authors"];
      const header = cols.map((col) => `<th>${col}</th>`).join("");
      const rows = tablePayload.rows
        .map((row) => {
          const cells = cols
            .map((col) => {
              const value = row[col] ?? "";
              if (col === "link" && value) {
                return `<td><a href="${value}" target="_blank" rel="noreferrer">${value}</a></td>`;
              }
              return `<td>${value}</td>`;
            })
            .join("");
          return `<tr>${cells}</tr>`;
        })
        .join("");
      table.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
    }

    runButton.addEventListener("click", async () => {
      const maxResults = Number(document.getElementById("max_results").value || 10);
      const downloadCount = Number(document.getElementById("download_count").value || 1);
      const runName = document.getElementById("run_name").value.trim() || null;
      const outputRoot = document.getElementById("output_root").value.trim() || "agentic_outputs";
      const payload = { output_root: outputRoot, run_name: runName };
      payload.brief = document.getElementById("brief").value.trim();
      payload.max_results = maxResults;
      payload.downloads = downloadCount;
      payload.min_queries = Number(document.getElementById("min_queries").value || 4);
      payload.max_queries = Number(document.getElementById("max_queries").value || 12);
      const targetPapers = document.getElementById("target_papers").value.trim();
      payload.target_papers = targetPapers ? Number(targetPapers) : null;
      payload.require_primary_anchor = document.getElementById("require_primary_anchor").value === "true";
      payload.model = document.getElementById("model").value.trim() || "gpt-4o-mini";
      payload.config_path = document.getElementById("config_path").value.trim() || "agentic_config.yaml";
      const endpoint = "/crawl/agentic";
      if (!payload.brief) {
        setStatus("error", "Agentic brief is required.");
        return;
      }

      setStatus("running", "Crawler running...");
      runButton.disabled = true;
      output.textContent = "{}";
      summary.innerHTML = "";
      table.innerHTML = "";

      try {
        const resp = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || "Request failed");
        }
        output.textContent = JSON.stringify(data, null, 2);
        const search = (data.result || {}).search || {};
        const downloads = (data.result || {}).downloads || {};
        const lines = [
          "Total records: " + (search.total_records ?? 0),
          "Primary anchor: " + (search.primary_anchor || "-"),
          "Metadata path: " + (search.metadata_path || "-"),
          "Download dir: " + (downloads.download_dir || "-"),
          "Summary path: " + (data.result || {}).summary_path || "-"
        ];
        summary.innerHTML = "<pre>" + lines.join("\\n") + "</pre>";
        renderTable(data.table);
        setStatus("ok", "Done. Results written to " + data.run_dir);
      } catch (err) {
        output.textContent = JSON.stringify({ error: String(err) }, null, 2);
        summary.innerHTML = "";
        table.innerHTML = "";
        setStatus("error", "Error running crawl.");
      } finally {
        runButton.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.post("/crawl/simple", response_model=SimpleCrawlResponse)
def crawl_simple(payload: SimpleCrawlRequest) -> SimpleCrawlResponse:
    try:
        run_dir = _make_run_dir(payload.output_root, payload.run_name)
        result = run_simple(
            query=payload.query,
            max_results=payload.max_results,
            anchor=payload.anchor,
            download_count=payload.download_count,
            out_dir=run_dir,
        )
        table = _build_table(_load_metadata(result.get("metadata")))
        return SimpleCrawlResponse(run_dir=str(run_dir), result=result, table=table)
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _build_agentic_args(payload: AgenticCrawlRequest) -> Namespace:
    return Namespace(
        brief=payload.brief,
        max_results=payload.max_results,
        downloads=payload.downloads,
        min_queries=payload.min_queries,
        max_queries=payload.max_queries,
        target_papers=payload.target_papers,
        recall_cap=payload.recall_cap,
        precision_cap=payload.precision_cap,
        require_primary_anchor=payload.require_primary_anchor,
        model=payload.model,
        output=payload.output_root,
        config=payload.config_path,
        verbose=payload.verbose,
    )


@app.post("/crawl/agentic", response_model=AgenticCrawlResponse)
def crawl_agentic(payload: AgenticCrawlRequest) -> AgenticCrawlResponse:
    try:
        args = _build_agentic_args(payload)
        config = load_config(payload.config_path)
        outputs = run_agentic_pipeline(args, config)
        metadata_path = (outputs.get("search") or {}).get("metadata_path")
        table = _build_table(_load_metadata(metadata_path))
        return AgenticCrawlResponse(run_dir=outputs.get("run_dir", ""), result=outputs, table=table)
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/crawl/simple/stream")
def crawl_simple_stream(payload: SimpleCrawlRequest) -> StreamingResponse:
    def event_stream():
        yield json.dumps({"event": "start"}) + "\n"
        try:
            run_dir = _make_run_dir(payload.output_root, payload.run_name)
            result = run_simple(
                query=payload.query,
                max_results=payload.max_results,
                anchor=payload.anchor,
                download_count=payload.download_count,
                out_dir=run_dir,
            )
            table = _build_table(_load_metadata(result.get("metadata")))
            yield json.dumps(
                {"event": "done", "run_dir": str(run_dir), "result": result, "table": table.dict()}
            ) + "\n"
        except Exception as exc:
            yield json.dumps({"event": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")


@app.post("/crawl/agentic/stream")
def crawl_agentic_stream(payload: AgenticCrawlRequest) -> StreamingResponse:
    def event_stream():
        yield json.dumps({"event": "start"}) + "\n"
        try:
            args = _build_agentic_args(payload)
            config = load_config(payload.config_path)
            outputs = run_agentic_pipeline(args, config)
            metadata_path = (outputs.get("search") or {}).get("metadata_path")
            table = _build_table(_load_metadata(metadata_path))
            yield json.dumps(
                {"event": "done", "run_dir": outputs.get("run_dir", ""), "result": outputs, "table": table.dict()}
            ) + "\n"
        except Exception as exc:
            yield json.dumps({"event": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
