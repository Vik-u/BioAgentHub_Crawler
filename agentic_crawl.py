#!/usr/bin/env python3
"""CrewAI-driven multi-agent pipeline for BioAgentHub crawler."""

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from crewai import Agent, Crew, Process, Task
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from pypdf import PdfReader
import yaml

ROOT = Path(__file__).resolve().parent

from env_utils import load_env
from metadata_enrichment import (
    aggregate_relevance_stats,
    apply_hybrid_relevance_scoring,
    enrich_abstracts,
)

from pdf_downloader import ImprovedPDFDownloader
from simple_crawl import deduplicate
from tools import biorxiv_search_func, europepmc_search_func
from zotero_sync import get_collection_info, sync_papers_to_zotero, zotero_configured
from usage_tracker import UsageTracker


load_env()

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "than",
    "the",
    "to",
    "with",
}

METHOD_TERMS = [
    "saturation mutagenesis",
    "error prone pcr",
    "error-prone pcr",
    "directed evolution",
    "ancestral reconstruction",
    "structure guided design",
    "structure-guided design",
    "computational design",
    "machine learning guided",
    "rational design",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _basic_tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def _filtered_tokens(text: Optional[str]) -> List[str]:
    return [tok for tok in _basic_tokenize(text) if len(tok) > 2 and tok not in STOPWORDS]


def _collect_base_tokens(primary_anchor: Optional[str], queries: Sequence[Dict[str, Any]]) -> set:
    tokens: set = set()

    def add_text(block: Optional[str]) -> None:
        for tok in _filtered_tokens(block):
            tokens.add(tok)

    add_text(primary_anchor)
    for item in queries or []:
        add_text(item.get("query"))
        for anchor in item.get("supporting_anchors") or []:
            add_text(anchor)

    return tokens


def _extract_candidate_terms(
    papers: Sequence[Dict[str, Any]],
    base_tokens: set,
    used_phrases: set,
    max_candidates: int = 10,
) -> List[str]:
    tokens: List[str] = []
    text_blobs: List[str] = []
    for paper in papers:
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        combined = f"{title} {abstract}".strip()
        text_blobs.append(combined.lower())
        tokens.extend(_filtered_tokens(combined))

    if not tokens:
        return []

    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()
    for idx in range(len(tokens) - 1):
        window = tokens[idx : idx + 2]
        if any(tok in STOPWORDS for tok in window):
            continue
        if all(tok in base_tokens for tok in window):
            continue
        phrase = " ".join(window)
        if phrase in used_phrases:
            continue
        bigram_counts[phrase] += 1
    for idx in range(len(tokens) - 2):
        window = tokens[idx : idx + 3]
        if any(tok in STOPWORDS for tok in window):
            continue
        if all(tok in base_tokens for tok in window):
            continue
        phrase = " ".join(window)
        if phrase in used_phrases:
            continue
        trigram_counts[phrase] += 1

    candidates: List[Tuple[float, str]] = []
    for phrase, freq in bigram_counts.items():
        candidates.append((freq, phrase))
    for phrase, freq in trigram_counts.items():
        candidates.append((freq * 1.2, phrase))

    blob_text = " \n".join(text_blobs)
    for method in METHOD_TERMS:
        normalized = method.lower()
        if normalized in used_phrases:
            continue
        if normalized in blob_text and any(tok not in base_tokens for tok in normalized.split()):
            # Slight boost so strategy-focused terms bubble up even with low counts.
            score = blob_text.count(normalized) * 2 + len(normalized.split())
            candidates.append((max(score, 1.5), normalized))

    candidates.sort(key=lambda item: item[0], reverse=True)
    unique_phrases: List[str] = []
    for _, phrase in candidates:
        if phrase in used_phrases:
            continue
        unique_phrases.append(phrase)
        if len(unique_phrases) >= max_candidates:
            break
    return unique_phrases


def _build_expansion_queries(
    primary_anchor: Optional[str],
    candidate_phrases: Sequence[str],
    expansion_cap: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not primary_anchor:
        return [], []

    expansion_queries: List[Dict[str, Any]] = []
    phrases_used: List[str] = []
    for phrase in candidate_phrases[:3]:
        terms = phrase.split()
        if not terms:
            continue
        quoted_phrase = f'"{phrase}"' if len(terms) > 1 else terms[0]
        query_str = f"{primary_anchor} AND {quoted_phrase}"
        notes = f"Expansion query derived from harvested literature mentioning '{phrase}'."
        expansion_queries.append(
            {
                "query": query_str,
                "query_type": "precision",
                "natural_language_prompt": notes,
                "supporting_anchors": [primary_anchor, phrase],
                "notes": notes,
                "max_results": expansion_cap,
                "is_expansion": True,
            }
        )
        phrases_used.append(phrase)

    return expansion_queries, phrases_used


def _execute_single_query(
    item: Dict[str, Any],
    primary_anchor: Optional[str],
    require_primary_anchor: bool,
    recall_cap: int,
    precision_cap: int,
    per_query_cap: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    query = (item.get("query") or "").strip()
    anchor_override = item.get("anchor")
    anchor = primary_anchor if require_primary_anchor or not anchor_override else anchor_override
    query_type = (item.get("query_type") or "precision").lower()
    is_expansion = bool(item.get("is_expansion"))
    supporting_anchors = item.get("supporting_anchors") or []
    notes = item.get("notes")

    default_cap = recall_cap if query_type == "recall" else precision_cap
    if per_query_cap:
        default_cap = min(default_cap, per_query_cap)
    desired_cap = item.get("max_results") or default_cap
    per_cap = min(desired_cap, default_cap)

    stat = {
        "query": query,
        "anchor": anchor,
        "supporting_anchors": supporting_anchors,
        "notes": notes,
        "query_type": query_type,
        "retrieved": 0,
        "is_expansion": is_expansion,
        "cap": per_cap,
    }

    if not query:
        return [], stat

    hits: List[Dict[str, Any]] = []
    hits.extend(europepmc_search_func(query, max_results=per_cap, anchor=anchor))
    hits.extend(biorxiv_search_func(query, max_results=per_cap, anchor=anchor))
    unique_hits = deduplicate(hits)
    for record in unique_hits:
        record["query_used"] = query
        record["anchor_used"] = anchor
        record["supporting_anchors"] = supporting_anchors
        record["notes"] = notes
        record["query_type"] = query_type
        record["is_expansion_query"] = is_expansion
    stat["retrieved"] = len(unique_hits)
    return unique_hits, stat


def _execute_query_batch(
    queries: Sequence[Dict[str, Any]],
    primary_anchor: Optional[str],
    require_primary_anchor: bool,
    recall_cap: int,
    precision_cap: int,
    per_query_cap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    aggregated: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []
    for item in queries or []:
        hits, stat = _execute_single_query(
            item,
            primary_anchor,
            require_primary_anchor,
            recall_cap,
            precision_cap,
            per_query_cap,
        )
        if hits:
            aggregated.extend(hits)
        stats.append(stat)
    return aggregated, stats


def _run_dynamic_expansion_loop(
    aggregated: List[Dict[str, Any]],
    per_query_stats: List[Dict[str, Any]],
    base_tokens: set,
    primary_anchor: Optional[str],
    target_results: Optional[int],
    recall_cap: int,
    precision_cap: int,
    per_query_cap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not primary_anchor or not target_results:
        return aggregated, per_query_stats, []

    expansion_rounds: List[Dict[str, Any]] = []
    used_phrases: set = set()
    max_rounds = 5
    expansion_cap = max(5, min(20, precision_cap if precision_cap else 20))

    round_idx = 0
    while len(aggregated) < target_results and round_idx < max_rounds:
        round_idx += 1
        enrich_abstracts(aggregated)
        candidate_terms = _extract_candidate_terms(aggregated, base_tokens, used_phrases)
        if not candidate_terms:
            break
        expansion_queries, phrases_used = _build_expansion_queries(primary_anchor, candidate_terms, expansion_cap)
        if not expansion_queries:
            break
        used_phrases.update(phrases_used)
        hits, stats = _execute_query_batch(
            expansion_queries,
            primary_anchor,
            True,
            recall_cap,
            precision_cap,
            per_query_cap,
        )
        prev_total = len(aggregated)
        if hits:
            aggregated = deduplicate(aggregated + hits)
            enrich_abstracts(aggregated)
        added = len(aggregated) - prev_total
        per_query_stats.extend(stats)
        expansion_rounds.append(
            {
                "round": round_idx,
                "expansion_queries": [
                    {
                        "query": q["query"],
                        "query_type": q.get("query_type", "precision"),
                        "supporting_anchors": q.get("supporting_anchors", []),
                        "notes": q.get("notes"),
                    }
                    for q in expansion_queries
                ],
                "new_unique_records": added,
            }
        )

        for phrase in phrases_used:
            base_tokens.update(_filtered_tokens(phrase))

        if added <= 0:
            break
        increment = added / max(1, prev_total)
        if increment < 0.05:
            break

    return aggregated, per_query_stats, expansion_rounds


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set in your environment for CrewAI agents.")


def _parse_json_block(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def agent_temperature(config: Dict[str, Any], agent_key: str, default: float) -> float:
    return (
        config.get("agents", {})
        .get(agent_key, {})
        .get("temperature", default)
    )


def make_llm_builder(config: Dict[str, Any], default_model: Optional[str], tracker: Optional[UsageTracker] = None):
    llm_cfg = config.get("llm", {})
    provider = (llm_cfg.get("provider") or "openai").lower()

    if provider == "ollama":
        model = llm_cfg.get("model") or default_model or "llama3"
        base_url = llm_cfg.get("base_url")

        def builder(temp: float):
            return ChatOllama(model=model, base_url=base_url, temperature=temp)

        return builder, provider

    model_name = llm_cfg.get("model") or default_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def builder(temp: float):
        if tracker:
            return ChatOpenAI(model=model_name, temperature=temp, callbacks=[tracker])
        return ChatOpenAI(model=model_name, temperature=temp)

    return builder, "openai"


def _literature_search_tool(tool_input: str) -> str:
    payload = json.loads(tool_input)
    queries = payload.get("queries", [])
    metadata_path = Path(payload.get("metadata_path", "agentic_outputs/papers.json"))
    per_query_cap = payload.get("per_query_cap", 6)
    global_cap = payload.get("global_cap")
    primary_anchor = payload.get("primary_anchor")
    require_primary_anchor = payload.get("require_primary_anchor", False)
    target_results = payload.get("target_results")
    recall_cap = payload.get("recall_cap", per_query_cap)
    precision_cap = payload.get("precision_cap", per_query_cap)
    user_brief = payload.get("user_brief")
    scoring_cfg = payload.get("scoring") or {}

    aggregated, per_query_stats = _execute_query_batch(
        queries,
        primary_anchor,
        require_primary_anchor,
        recall_cap,
        precision_cap,
        per_query_cap,
    )
    aggregated = deduplicate(aggregated)
    enrich_abstracts(aggregated)

    base_tokens = _collect_base_tokens(primary_anchor, queries)
    aggregated, per_query_stats, expansion_rounds = _run_dynamic_expansion_loop(
        aggregated,
        per_query_stats,
        base_tokens,
        primary_anchor,
        target_results,
        recall_cap,
        precision_cap,
        per_query_cap,
    )

    if target_results and primary_anchor and len(aggregated) < target_results:
        deficit = max(0, target_results - len(aggregated))
        if deficit:
            fallback_hits = europepmc_search_func(primary_anchor, max_results=deficit, anchor=None)
            fallback_hits = deduplicate(fallback_hits)
            for record in fallback_hits:
                record["query_used"] = f"fallback:{primary_anchor}"
                record["anchor_used"] = primary_anchor
                record["supporting_anchors"] = [primary_anchor]
                record["query_type"] = "recall"
                record["is_expansion_query"] = False
            aggregated.extend(fallback_hits)
            aggregated = deduplicate(aggregated)
            per_query_stats.append(
                {
                    "query": f"fallback:{primary_anchor}",
                    "anchor": None,
                    "supporting_anchors": [primary_anchor],
                    "notes": "Automatic fallback to anchor-only search.",
                    "query_type": "recall",
                    "retrieved": len(fallback_hits),
                    "is_expansion": False,
                    "cap": deficit,
                }
            )

    enrich_abstracts(aggregated)
    scoring_metadata = apply_hybrid_relevance_scoring(
        aggregated,
        primary_anchor,
        queries,
        user_brief=user_brief,
        scoring_config=scoring_cfg,
    )

    if global_cap:
        aggregated = aggregated[:global_cap]

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    sample_payload = []
    for record in aggregated[: min(10, len(aggregated))]:
        sample_payload.append(
            {
                "title": record.get("title"),
                "source": record.get("source"),
                "doi": record.get("doi"),
                "pmid": record.get("pmid"),
                "published": record.get("published"),
                "abstract": (record.get("abstract") or "No abstract")[:600],
                "query_used": record.get("query_used"),
                "supporting_anchors": record.get("supporting_anchors", []),
                "relevance_score": record.get("relevance_score"),
                "query_type": record.get("query_type"),
                "is_expansion_query": record.get("is_expansion_query", False),
                "token_score": record.get("token_score"),
                "embed_score": record.get("embed_score"),
                "cross_encoder_score": record.get("cross_encoder_score"),
                "final_score": record.get("final_score"),
            }
        )

    response = {
        "metadata_path": str(metadata_path),
        "total_records": len(aggregated),
        "primary_anchor": primary_anchor,
        "require_primary_anchor": require_primary_anchor,
        "target_results": target_results,
        "per_query": per_query_stats,
        "expansion_rounds": expansion_rounds,
        "sample_papers": sample_payload,
        "relevance_stats": aggregate_relevance_stats(aggregated),
        "scoring_metadata": scoring_metadata,
    }
    return json.dumps(response)


def _download_pdfs_tool(tool_input: str) -> str:
    payload = json.loads(tool_input)
    metadata_path = Path(payload["metadata_path"])
    download_dir = Path(payload.get("download_dir", metadata_path.parent / "pdfs"))
    max_downloads = payload.get("max_downloads", 2)

    with open(metadata_path) as f:
        papers: List[Dict[str, Any]] = json.load(f)

    download_dir.mkdir(parents=True, exist_ok=True)
    downloader = ImprovedPDFDownloader()
    logs: List[Dict[str, Any]] = []
    for idx, paper in enumerate(papers[:max_downloads], start=1):
        logs.append(downloader.download_paper(paper, download_dir, idx))

    log_path = download_dir / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    response = {
        "log_path": str(log_path),
        "download_dir": str(download_dir),
        "attempted": min(max_downloads, len(papers)),
        "success": sum(1 for entry in logs if entry.get("status") == "success"),
        "files": [entry.get("filename") for entry in logs if entry.get("filename")],
    }
    return json.dumps(response)


def _pdf_text_tool(tool_input: str) -> str:
    payload = json.loads(tool_input)
    pdf_dir = Path(payload["pdf_dir"])
    filenames = payload.get("filenames") or []
    max_chars = payload.get("max_chars", 8000)

    excerpts = []
    missing = []

    for fname in filenames:
        pdf_path = pdf_dir / fname
        if not pdf_path.exists():
            missing.append(fname)
            continue
        try:
            reader = PdfReader(str(pdf_path))
            text_chunks = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_chunks.append(page_text.strip())
                joined = "\n".join(text_chunks)
                if len(joined) >= max_chars:
                    break
            excerpt = "\n".join(text_chunks)[:max_chars]
            excerpts.append({"filename": fname, "text": excerpt})
        except Exception as exc:
            missing.append(f"{fname}: {exc}")

    return json.dumps({
        "pdf_dir": str(pdf_dir),
        "excerpts": excerpts,
        "missing": missing,
    })


LITERATURE_SEARCH = Tool(
    name="literature_search",
    func=_literature_search_tool,
    description=(
        "Use to execute BioAgentHub crawler searches. Pass JSON with 'primary_anchor', 'queries', \n"
        "'metadata_path', 'per_query_cap', 'global_cap', 'require_primary_anchor', 'target_results', \n"
        "'recall_cap', 'precision_cap', 'user_brief', and optional 'scoring'. Returns JSON summary and writes papers.json."
    ),
)

PDF_FETCH = Tool(
    name="pdf_fetcher",
    func=_download_pdfs_tool,
    description="Download PDFs for the selected papers. Input JSON must include 'metadata_path', 'download_dir', and 'max_downloads'.",
)

PDF_TEXT = Tool(
    name="pdf_text_reader",
    func=_pdf_text_tool,
    description="Extracts raw text excerpts from downloaded PDFs. Input JSON must include 'pdf_dir', plus optional 'filenames' and 'max_chars'.",
)


def build_agent(
    role: str,
    goal: str,
    backstory: str,
    temperature: float,
    tools=None,
    verbose: bool = False,
    llm_builder=None,
) -> Agent:
    if llm_builder:
        llm = llm_builder(temperature)
    else:
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        memory=False,
        tools=tools or [],
        max_iter=5,
    )


def run_agentic_pipeline(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.perf_counter()
    usage_tracker = UsageTracker()
    llm_builder, provider = make_llm_builder(config, args.model, tracker=usage_tracker)
    if provider == "openai":
        _ensure_api_key()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / f"agentic_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "papers_agentic.json"
    download_dir = run_dir / "pdfs"
    target_papers = args.target_papers or config.get("search", {}).get("target_results")
    scoring_config = config.get("scoring", {})
    scoring_payload = json.dumps(scoring_config)
    brief_literal = json.dumps(args.brief)

    gatekeeper = None
    if zotero_configured() and args.gatekeeper != "off":
        gatekeeper = get_collection_info(args.brief, args.zotero_collection)
        if gatekeeper and gatekeeper.get("exists") and args.gatekeeper == "skip" and gatekeeper.get("num_items", 0) > 0:
            outputs = {
                "skipped": True,
                "reason": "collection_exists",
                "gatekeeper": gatekeeper,
                "run_dir": str(run_dir),
                "usage": usage_tracker.summary(),
                "stats": {"runtime_sec": round(time.perf_counter() - start_time, 2)},
            }
            summary_path = run_dir / "agentic_summary.json"
            with open(summary_path, "w") as f:
                json.dump(outputs, f, indent=2)
            outputs["summary_path"] = str(summary_path)
            return outputs

    query_agent = build_agent(
        role="Query Expansion Specialist",
        goal="Transform a single research brief into multiple anchored queries for PETase literature discovery.",
        backstory="Enzyme engineer trained to translate desired traits into precise literature search queries.",
        temperature=agent_temperature(config, "query", 0.35),
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    search_agent = build_agent(
        role="Literature Search Orchestrator",
        goal="Run BioAgentHub crawler tools to retrieve the most relevant PETase papers for downstream agents.",
        backstory="Veteran bioinformatician who knows how to squeeze maximum value from Europe PMC and bioRxiv.",
        temperature=agent_temperature(config, "search", 0.1),
        tools=[LITERATURE_SEARCH],
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    pdf_agent = build_agent(
        role="Open Access Downloader",
        goal="Guarantee that the most relevant references have local PDFs ready for review.",
        backstory="Digital librarian skilled at navigating PMC, Europe PMC, Unpaywall, and DOI fallbacks.",
        temperature=agent_temperature(config, "pdf", 0.1),
        tools=[PDF_FETCH],
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    query_rank_agent = build_agent(
        role="Query Strategy Analyst",
        goal="Prioritize search queries that best satisfy the research brief using downstream evidence.",
        backstory="Competitive intelligence lead who constantly reallocates search effort based on observed payoffs.",
        temperature=agent_temperature(config, "query_rerank", 0.2),
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    pdf_rerank_agent = build_agent(
        role="PDF Evidence Reranker",
        goal="Read downloaded PDFs and reorder papers by strength of evidence for the design brief.",
        backstory="Principal scientist who verifies claims by inspecting full-text experiments before approving them.",
        temperature=agent_temperature(config, "pdf_rerank", 0.1),
        tools=[PDF_TEXT],
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    summarizer_agent = build_agent(
        role="Evidence Summarizer",
        goal="Synthesize crawler outputs into actionable PETase engineering insights and statistics.",
        backstory="Polymer biodegradation scientist charged with briefing stakeholders.",
        temperature=agent_temperature(config, "summarizer", 0.2),
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    validator_agent = build_agent(
        role="Relevancy Validator",
        goal="Score the collected papers for how well they address the research brief.",
        backstory="Critical reviewer tasked with ranking evidence quality and focus.",
        temperature=agent_temperature(config, "validator", 0.0),
        verbose=args.verbose,
        llm_builder=llm_builder,
    )

    query_task = Task(
        description=dedent(
            f"""
            Expand the research brief into targeted search plans.

            Brief:
            {args.brief}

            Steps for the Query Expansion Specialist:
              1. Identify a single PRIMARY anchor keyword (usually the core target from the brief) that should appear across every query unless you explicitly note why an exception is required.
              2. Brainstorm as many supporting angles as needed and generate between {args.min_queries} and {args.max_queries} query objects (you may exceed the max only if vital coverage is missing). Stop when the demand of the brief feels saturated.
              3. Each query object must include:
                   - query: a compact Boolean-style search string (use quotes, AND/OR, truncation) centered on the PRIMARY anchor plus supporting tokens. Avoid long sentences.
                   - query_type: either "recall" (broad coverage, <=1 extra concept) or "precision" (2-4 constraints emphasizing methods/outcomes). Use at least one of each.
                   - natural_language_prompt: one sentence describing the human-readable intent (optional but recommended).
                   - supporting_anchors: list of 1-3 short phrases (synonyms/aliases/method keywords) that motivated the query.
                   - notes: rationale describing what evidence this query should surface or why it is unique.
                   - max_results: set to {args.recall_cap} for recall queries and {args.precision_cap} for precision queries.

            Ensure the collection covers complementary angles (protein engineering, mutagenesis, structure prediction, assay benchmarking, host optimisation, robustness studies, expression hosts, guided evolution, etc.).
            Output JSON must expose two top-level keys: `primary_anchor` (string) and `queries` (list of the defined objects including query_type and supporting_anchors).
            """
        ).strip(),
        agent=query_agent,
        expected_output="JSON with keys primary_anchor and queries (each query includes query, query_type, natural_language_prompt, supporting_anchors, notes, max_results)",
    )

    search_task = Task(
        description=dedent(
            f"""
            Use the JSON from the query expansion task as your sole input.
            Steps (default require_primary_anchor={args.require_primary_anchor}):
              1. Parse the JSON carefully.
              2. Retrieve `primary_anchor` and determine whether to enforce it strictly. If the context already looks narrow, set require_primary_anchor to False so the search can explore supporting anchors more freely.
              3. Call the `literature_search` tool exactly once with a JSON string containing the keys:
                 primary_anchor, require_primary_anchor (bool), queries (preserving query_type/supporting_anchors/max_results), metadata_path="{metadata_path}", per_query_cap={args.max_results}, global_cap={args.max_results}, target_results={target_papers or args.max_results}, recall_cap={args.recall_cap}, precision_cap={args.precision_cap}, user_brief={brief_literal}, scoring={scoring_payload}
              4. The tool response already contains metadata; return that exact JSON and add an 'insights' field summarizing the strongest hits in <=4 bullet sentences.
            Output must remain valid JSON.
            """
        ).strip(),
        agent=search_agent,
        context=[query_task],
        expected_output="JSON with keys metadata_path, primary_anchor, require_primary_anchor, total_records, per_query, sample_papers, insights",
    )

    download_task = Task(
        description=dedent(
            f"""
            Download PDFs for the highest priority hits.
            - Use the metadata_path provided by the search task.
            - Call `pdf_fetcher` exactly once with JSON {{"metadata_path": "{metadata_path}", "download_dir": "{download_dir}", "max_downloads": {args.downloads}}}
            - Return the raw tool JSON and add a 'note' field describing failures or follow-ups.
            Output must be JSON.
            """
        ).strip(),
        agent=pdf_agent,
        context=[search_task],
        expected_output="JSON with log_path, download_dir, attempted, success, files, note",
    )

    summary_task = Task(
        description=dedent(
            """
            Write an executive summary for scientists engineering PETase enzymes for improved activity, thermostability, and robustness.
            Use every prior artifact (query plan, search stats, download log).
            Respond in JSON with:
              - overview: concise paragraph
              - stats: dictionary covering counts, anchors, sources, downloads
              - key_findings: list of bullet strings referencing paper titles or DOIs
              - next_steps: list of concrete follow-up actions
            """
        ).strip(),
        agent=summarizer_agent,
        context=[query_task, search_task, download_task],
        expected_output="JSON with overview, stats, key_findings, next_steps",
    )

    query_rerank_task = Task(
        description=dedent(
            """
            Re-rank the query expansion set using observed outcomes.
            - Ingest the original query plan, search stats (per-query retrieval), and executive summary insights.
            - Score each query object on a 0-1 scale for utility (consider recall, uniqueness, downstream usefulness) and provide concise reasoning.
            - Produce JSON: {{"query_ranking": [ {{"query": str, "score": float, "reasoning": str}} ], "recommendations": ["how to adjust future queries"]}} sorted by score.
            """
        ).strip(),
        agent=query_rank_agent,
        context=[query_task, search_task, summary_task],
        expected_output="JSON with query_ranking and recommendations",
    )

    pdf_rerank_task = Task(
        description=dedent(
            f"""
            Use downloaded PDFs (if any) plus prior summaries to re-rank the literature.
            Instructions:
              - Inspect the search metadata and download log to identify available filenames under {download_dir}.
              - If at least one PDF exists, call the `pdf_text_reader` tool with JSON {{"pdf_dir": "{download_dir}", "filenames": TARGET_FILES, "max_chars": 10000}} to fetch excerpts. Use multiple calls if needed.
              - Combine PDF evidence with metadata/summary context to assign an evidence strength score (0-1) and justification for each paper (limit to top 10 papers; if fewer PDFs, fall back to metadata).
              - Output JSON: {{"paper_ranking": [ {{"title": str, "identifier": str, "score": float, "evidence": str}} ], "notes": str}}.
            """
        ).strip(),
        agent=pdf_rerank_agent,
        context=[search_task, download_task, summary_task, query_rerank_task],
        expected_output="JSON with paper_ranking and notes",
    )

    validator_task = Task(
        description=dedent(
            f"""
            Validate relevancy against the research brief: {args.brief}.
            Score each sample paper (cap at 6) with a relevance value between 0 and 1 and include short reasoning.
            Return JSON:
              {{
                "ranking": [{{"title": str, "doi_or_pmid": str, "relevance": float, "reasoning": str}}],
                "top_recommendations": [str]
              }}
            Ensure ranking sorted by relevance descending.
            """
        ).strip(),
        agent=validator_agent,
        context=[search_task, summary_task, query_rerank_task, pdf_rerank_task],
        expected_output="JSON with ranking and top_recommendations",
    )

    crew = Crew(
        agents=[query_agent, search_agent, pdf_agent, summarizer_agent, query_rank_agent, pdf_rerank_agent, validator_agent],
        tasks=[
            query_task,
            search_task,
            download_task,
            summary_task,
            query_rerank_task,
            pdf_rerank_task,
            validator_task,
        ],
        process=Process.sequential,
        verbose=2 if args.verbose else 0,
    )

    final_result = crew.kickoff()

    outputs = {
        "query_plan": _parse_json_block(query_task.output.result),
        "search": _parse_json_block(search_task.output.result),
        "downloads": _parse_json_block(download_task.output.result),
        "summary": _parse_json_block(summary_task.output.result),
        "query_rerank": _parse_json_block(query_rerank_task.output.result),
        "pdf_rerank": _parse_json_block(pdf_rerank_task.output.result),
        "validation": _parse_json_block(validator_task.output.result),
        "final_message": final_result,
        "run_dir": str(run_dir),
    }

    zotero_result = None
    if zotero_configured() and not args.no_zotero:
        try:
            papers = json.loads(metadata_path.read_text())
        except Exception:
            papers = []
        download_log = outputs.get("downloads", {}).get("log_path")
        download_log_path = Path(download_log) if download_log else download_dir / "download_log.json"
        zotero_result = sync_papers_to_zotero(
            papers,
            topic=args.brief,
            download_log_path=download_log_path,
            pdf_dir=download_dir,
            collection_override=args.zotero_collection,
            attach_pdfs=args.with_pdf,
            dedupe_local=not args.no_zotero_local_dedupe,
            dedupe_remote=not args.no_zotero_remote_dedupe,
        )

    outputs["usage"] = usage_tracker.summary()
    outputs["stats"] = {"runtime_sec": round(time.perf_counter() - start_time, 2)}
    outputs["zotero"] = zotero_result

    summary_path = run_dir / "agentic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(outputs, f, indent=2)
    outputs["summary_path"] = str(summary_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrewAI multi-agent orchestration for BioAgentHub crawler.")
    parser.add_argument("--brief", required=True, help="High-level research request (e.g., engineered PETase objective).")
    parser.add_argument("--max-results", dest="max_results", type=int, default=8, help="Max records to keep after dedupe.")
    parser.add_argument("--downloads", type=int, default=2, help="How many PDFs to attempt.")
    parser.add_argument("--min-queries", type=int, default=4, help="Minimum number of query variants the agent should target.")
    parser.add_argument("--max-queries", type=int, default=12, help="Maximum number of query variants before the agent must stop unless coverage is still missing.")
    parser.add_argument("--target-papers", type=int, default=None, help="Preferred minimum number of unique papers to gather before truncation.")
    parser.add_argument("--recall-cap", type=int, default=40, help="Per-query cap (max_results) for recall-tier queries.")
    parser.add_argument("--precision-cap", type=int, default=15, help="Per-query cap (max_results) for precision-tier queries.")
    parser.add_argument("--require-primary-anchor", action="store_true", help="Force every crawler call to enforce the primary anchor as a hard filter.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI chat model for agents.")
    parser.add_argument("--output", default="agentic_outputs", help="Directory for agent outputs.")
    parser.add_argument("--config", default="agentic_config.yaml", help="Path to YAML config file (optional).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose CrewAI logs.")
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = run_agentic_pipeline(args, config)
    print("Agentic pipeline complete. Key artifacts:")
    print(json.dumps({
        "run_dir": outputs["run_dir"],
        "metadata_path": outputs["search"].get("metadata_path"),
        "download_dir": outputs["downloads"].get("download_dir"),
        "summary_path": outputs["summary_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
