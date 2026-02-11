#!/usr/bin/env python3
"""Offline trainer for the hybrid relevance scoring weights (theta).

Reads labeled scoring components (token, embedding, cross-encoder) and
optimizes the softmax-parameterized blend used by metadata_enrichment.apply_hybrid_relevance_scoring.

Usage example:
    python learn_weights.py --data data/relevance_labels.jsonl --epochs 200 --lr 0.05

Supported input formats:
    - JSONL / NDJSON: each line is a JSON object with keys
        token_score, embed_score, cross_encoder_score, label (0 or 1), optional weight.
    - CSV: must include the same columns (header row required).

Results are written to scoring_weights.json by default (overridable via --out).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train softmax weights for hybrid scoring.")
    parser.add_argument("--data", required=True, help="Path to JSONL/NDJSON or CSV file with labeled samples.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of gradient-descent epochs (default: 200).")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for gradient descent (default: 0.05).")
    parser.add_argument("--out", default="scoring_weights.json", help="Output JSON file to store learned theta.")
    parser.add_argument("--init", default=None, help="Optional JSON file with initial theta values.")
    parser.add_argument("--clip", type=float, default=5.0, help="Gradient clipping value to avoid exploding grads.")
    parser.add_argument("--min-samples", type=int, default=10, help="Fail if fewer samples than this.")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch metrics.")
    return parser.parse_args()


def _softmax(theta: Iterable[float]) -> List[float]:
    values = list(theta)
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    denom = sum(exps) or 1.0
    return [val / denom for val in exps]


def _sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def load_initial_theta(path: str | None) -> List[float]:
    if not path:
        return [0.0, 0.0, 0.0]
    candidate = Path(path)
    if not candidate.exists():
        return [0.0, 0.0, 0.0]
    with open(candidate, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        theta = data.get("theta") or data.get("weights")
    else:
        theta = data
    if isinstance(theta, list) and len(theta) == 3:
        return [float(x) for x in theta]
    raise ValueError("Initial weights file must contain three numbers (theta).")


def load_samples(path: str) -> List[Dict[str, float]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    suffix = file_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson", ".json"}:
        return _load_jsonl(file_path)
    return _load_csv(file_path)


def _normalize_sample(sample: Dict[str, float]) -> Dict[str, float]:
    def clamp(value: float) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))

    normalized = {
        "token_score": clamp(sample.get("token_score", 0.0)),
        "embed_score": clamp(sample.get("embed_score", 0.0)),
        "cross_encoder_score": clamp(sample.get("cross_encoder_score", 0.0)),
        "label": clamp(sample.get("label", 0.0)),
        "weight": float(sample.get("weight", 1.0)),
    }
    return normalized


def _load_jsonl(path: Path) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            samples.append(_normalize_sample(raw))
    return samples


def _load_csv(path: Path) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(_normalize_sample(row))
    return samples


def _compute_loss_and_grad(samples: List[Dict[str, float]], theta: List[float]) -> Tuple[float, List[float]]:
    weights = _softmax(theta)
    grad = [0.0, 0.0, 0.0]
    total_loss = 0.0
    total_weight = 0.0
    for sample in samples:
        features = [
            sample["token_score"],
            sample["embed_score"],
            sample["cross_encoder_score"],
        ]
        label = sample["label"]
        weight = max(sample.get("weight", 1.0), 1e-6)
        score = sum(w * f for w, f in zip(weights, features))
        pred = _sigmoid(score)
        loss = -(label * math.log(pred + 1e-12) + (1 - label) * math.log(1 - pred + 1e-12))
        total_loss += loss * weight
        total_weight += weight

        error = (pred - label) * weight
        s_val = sum(error * features[i] * weights[i] for i in range(3))
        for k in range(3):
            g_w = error * features[k]
            grad[k] += weights[k] * (g_w - s_val)

    if total_weight:
        total_loss /= total_weight
        grad = [g / total_weight for g in grad]
    return total_loss, grad


def train_theta(samples: List[Dict[str, float]], theta: List[float], epochs: int, lr: float, clip: float, verbose: bool) -> List[float]:
    for epoch in range(1, epochs + 1):
        loss, grad = _compute_loss_and_grad(samples, theta)
        for idx in range(3):
            g = max(-clip, min(clip, grad[idx]))
            theta[idx] -= lr * g
        if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
            weights = _softmax(theta)
            print(f"Epoch {epoch:04d}: loss={loss:.4f}, theta={theta}, weights={weights}")
    return theta


def save_theta(theta: List[float], out_path: str) -> None:
    payload = {
        "theta": [round(val, 6) for val in theta],
        "weights": [round(val, 6) for val in _softmax(theta)],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved learned weights to {out_path}")


def main() -> None:
    args = parse_args()
    samples = load_samples(args.data)
    if len(samples) < args.min_samples:
        raise ValueError(f"Need at least {args.min_samples} samples; found {len(samples)}")
    theta = load_initial_theta(args.init) if args.init else load_initial_theta(None)
    theta = train_theta(samples, theta, args.epochs, args.lr, args.clip, args.verbose)
    save_theta(theta, args.out)


if __name__ == "__main__":
    main()
