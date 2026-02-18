"""Utilities for enriching literature metadata (abstracts, relevancy metrics)."""

from __future__ import annotations

import json
import math
import os
import re
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CROSSREF_URL = "https://api.crossref.org/works/"

DEFAULT_SCORING_WEIGHTS = Path(__file__).resolve().parent / "scoring_weights.json"
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DEFAULT_CROSS_ENCODER_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CROSS_ENCODER_TOPN = 100
EMBED_TEXT_LIMIT = 4000


def fetch_abstract_from_europepmc(doi: str, timeout: int = 15) -> Optional[str]:
    if not doi:
        return None
    params = {"query": f'DOI:"{doi}"', "format": "json", "pageSize": 1}
    response = requests.get(EUROPE_PMC_URL, params=params, timeout=timeout)
    if response.status_code != 200:
        return None
    results = response.json().get("resultList", {}).get("result", [])
    if not results:
        return None
    abstract = results[0].get("abstractText")
    return abstract


def fetch_abstract_from_crossref(doi: str, timeout: int = 15) -> Optional[str]:
    if not doi:
        return None
    response = requests.get(CROSSREF_URL + requests.utils.quote(doi), timeout=timeout)
    if response.status_code != 200:
        return None
    abstract = response.json().get("message", {}).get("abstract")
    if abstract:
        abstract = re.sub(r"<[^>]+>", " ", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract or None


def enrich_abstracts(papers: List[Dict]) -> None:
    for paper in papers:
        abstract = paper.get("abstract") or ""
        if abstract and abstract.lower() != "no abstract available":
            continue
        doi = (paper.get("doi") or "").strip()
        if not doi:
            continue
        fetched = fetch_abstract_from_europepmc(doi) or fetch_abstract_from_crossref(doi)
        if fetched:
            paper["abstract"] = fetched


WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _safe_json_loads(raw: str) -> Dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _get_openai_client() -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    try:
        return OpenAI()
    except Exception:
        return None


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def compute_relevance_score(
    paper: Dict,
    primary_anchor: Optional[str],
    query: str,
    supporting_anchors: Iterable[str],
) -> Dict:
    target_terms = set(_tokenize(query))
    if primary_anchor:
        target_terms.update(_tokenize(primary_anchor))
    for anchor in supporting_anchors or []:
        target_terms.update(_tokenize(anchor))
    target_terms = {term for term in target_terms if term and len(term) > 2}

    corpus = " ".join(
        filter(
            None,
            [paper.get("title"), paper.get("abstract"), paper.get("journal"), paper.get("category")],
        )
    ).lower()

    matched = {}
    for term in target_terms:
        count = corpus.count(term)
        if count:
            matched[term] = count

    if not target_terms:
        score = 0.0
    else:
        weighted_hits = sum(matched.values())
        score = min(1.0, weighted_hits / (len(target_terms) or 1))

    rounded = round(score, 4)
    return {
        "relevance_terms": sorted(matched.keys()),
        "relevance_score": rounded,
        "token_score": rounded,
        "relevance_hits": matched,
    }


def aggregate_relevance_stats(papers: List[Dict]) -> Dict:
    def _collect(field: str) -> List[float]:
        return [paper.get(field, 0.0) for paper in papers]

    def _stats(values: List[float]) -> Tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        avg = sum(values) / len(values)
        return round(avg, 4), round(max(values), 4), round(min(values), 4)

    token_stats = _stats(_collect("token_score"))
    embed_stats = _stats(_collect("embed_score"))
    cross_stats = _stats(_collect("cross_encoder_score"))
    final_stats = _stats(_collect("final_score"))

    summary = {
        "token": {"avg": token_stats[0], "max": token_stats[1], "min": token_stats[2]},
        "embedding": {"avg": embed_stats[0], "max": embed_stats[1], "min": embed_stats[2]},
        "cross_encoder": {"avg": cross_stats[0], "max": cross_stats[1], "min": cross_stats[2]},
        "final": {"avg": final_stats[0], "max": final_stats[1], "min": final_stats[2]},
    }
    summary["avg_score"] = final_stats[0]
    summary["max_score"] = final_stats[1]
    summary["min_score"] = final_stats[2]
    return summary


def _parse_theta(data: object) -> Optional[List[float]]:
    if isinstance(data, dict):
        theta = data.get("theta") or data.get("weights")
    else:
        theta = data
    if isinstance(theta, list) and len(theta) == 3:
        return [float(x) for x in theta]
    return None


def _load_packaged_weights() -> Optional[List[float]]:
    try:
        resource = resources.files("bioagenthub_crawler").joinpath("scoring_weights.json")
        if not resource.is_file():
            return None
        with resource.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _parse_theta(data)
    except Exception:
        return None


def load_scoring_weights(path: Optional[str]) -> List[float]:
    candidate = Path(path) if path else DEFAULT_SCORING_WEIGHTS
    if candidate.exists():
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        parsed = _parse_theta(data)
        if parsed is not None:
            return parsed
    packaged = _load_packaged_weights()
    if packaged is not None:
        return packaged
    return [0.0, 0.0, 0.0]


def _softmax(values: Sequence[float]) -> Tuple[float, float, float]:
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    denom = sum(exps) or 1.0
    weights = [val / denom for val in exps]
    return tuple(weights)  # type: ignore


def compute_final_score(
    token_score: float,
    embed_score: float,
    cross_encoder_score: float,
    theta: Sequence[float],
) -> Tuple[float, Tuple[float, float, float]]:
    weights = _softmax(theta)
    final_value = (
        weights[0] * token_score
        + weights[1] * embed_score
        + weights[2] * cross_encoder_score
    )
    return round(final_value, 4), weights


def compute_embedding(text: str, client: Optional[OpenAI], model: Optional[str]) -> Optional[List[float]]:
    if not client or not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    truncated = cleaned[:EMBED_TEXT_LIMIT]
    try:
        response = client.embeddings.create(model=model or DEFAULT_EMBED_MODEL, input=truncated)
        embedding = response.data[0].embedding
        return [float(x) for x in embedding]
    except Exception:
        return None


def _cosine_similarity(vec_a: Optional[Sequence[float]], vec_b: Optional[Sequence[float]]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _cross_encoder_prompt(brief: str, title: str, abstract: str) -> str:
    return (
        "You are a scientific relevance judge.\n"
        "Given the engineering brief and a paper title/abstract, return JSON with "
        "keys 'score' (0-1 float) and 'explanation' (<=25 words).\n"
        f"Brief: {brief.strip()}\n"
        f"Title: {title.strip() or 'N/A'}\n"
        f"Abstract: {abstract.strip() or 'N/A'}"
    )


def _run_cross_encoder(
    client: Optional[OpenAI],
    model: Optional[str],
    brief: Optional[str],
    papers: List[Dict],
    top_n: int,
) -> None:
    if not client or not model or not brief:
        return
    for paper in papers:
        token = paper.get("token_score", 0.0)
        embed = paper.get("embed_score", 0.0)
        paper["_provisional_score"] = token + embed
    ranked = sorted(papers, key=lambda p: p.get("_provisional_score", 0.0), reverse=True)
    for paper in ranked[:top_n]:
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        prompt = _cross_encoder_prompt(brief, title, abstract)
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = ""
            if getattr(response, "output", None):
                first = response.output[0]
                if getattr(first, "content", None):
                    chunk = first.content[0]
                    content = getattr(chunk, "text", "")
            if not content and getattr(response, "output_text", None):
                content = "\n".join(response.output_text)
            if not content:
                content = str(response)
            parsed = _safe_json_loads(content)
            score = float(parsed.get("score", 0.0))
            paper["cross_encoder_score"] = max(0.0, min(1.0, score))
            paper["cross_encoder_explanation"] = parsed.get("explanation")
        except Exception:
            paper.setdefault("cross_encoder_score", 0.0)
            paper.setdefault("cross_encoder_explanation", None)

    for paper in papers:
        paper.pop("_provisional_score", None)


def apply_hybrid_relevance_scoring(
    papers: List[Dict],
    primary_anchor: Optional[str],
    queries: Sequence[Dict],
    user_brief: Optional[str] = None,
    scoring_config: Optional[Dict] = None,
) -> Dict[str, Dict]:
    scoring_config = scoring_config or {}
    theta = scoring_config.get("theta") or load_scoring_weights(scoring_config.get("weights_path"))
    enable_embeddings = scoring_config.get("enable_embeddings", True)
    enable_cross = scoring_config.get("enable_cross_encoder", True)
    cross_top_n = int(scoring_config.get("cross_encoder_top_n", MAX_CROSS_ENCODER_TOPN))
    embed_model = scoring_config.get("embedding_model") or DEFAULT_EMBED_MODEL
    cross_model = scoring_config.get("cross_encoder_model") or DEFAULT_CROSS_ENCODER_MODEL

    client = None
    if (enable_embeddings or enable_cross) and user_brief:
        client = _get_openai_client()

    for paper in papers:
        paper.update(
            compute_relevance_score(
                paper,
                primary_anchor,
                paper.get("query_used", ""),
                paper.get("supporting_anchors", []),
            )
        )
    user_embedding = None
    if enable_embeddings and client and user_brief:
        user_embedding = compute_embedding(user_brief, client, embed_model)
    embedding_active = user_embedding is not None

    for paper in papers:
        embed_score = 0.0
        if user_embedding and enable_embeddings:
            text = " ".join(filter(None, [paper.get("title"), paper.get("abstract") or ""]))
            paper_embedding = compute_embedding(text, client, embed_model)
            similarity = _cosine_similarity(user_embedding, paper_embedding)
            embed_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        paper["embed_score"] = round(embed_score, 4)
        paper.setdefault("cross_encoder_score", 0.0)

    cross_active = False
    if enable_cross and user_brief and client:
        _run_cross_encoder(client, cross_model, user_brief, papers, cross_top_n)
        cross_active = True

    final_weights = None
    for paper in papers:
        final_score, weights = compute_final_score(
            paper.get("token_score", 0.0),
            paper.get("embed_score", 0.0),
            paper.get("cross_encoder_score", 0.0),
            theta,
        )
        paper["final_score"] = final_score
        paper["score_weights"] = {
            "token": round(weights[0], 4),
            "embedding": round(weights[1], 4),
            "cross_encoder": round(weights[2], 4),
        }
        if final_weights is None:
            final_weights = paper["score_weights"]

    if not final_weights:
        final_weights = {
            "token": round(1 / 3, 4),
            "embedding": round(1 / 3, 4),
            "cross_encoder": round(1 / 3, 4),
        }

    return {
        "weights": final_weights,
        "embedding_model": embed_model if embedding_active else None,
        "cross_encoder_model": cross_model if cross_active else None,
        "cross_encoder_top_n": min(cross_top_n, len(papers)) if cross_active else 0,
        "theta": theta,
    }
