#!/usr/bin/env python3
# 08b_bayes_postprocess_weights.py
# Read 08 infer json -> apply Bayesian shrinkage -> write bayes json (no model inference)

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _safe_float(x, default=None):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def load_infer_json(path: str) -> Tuple[List[str], Dict[str, float], Dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("infer_json must be a JSON object")

    selected = obj.get("selected", [])
    weights = obj.get("weights", {})

    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        raise ValueError("infer_json['selected'] must be list[str]")
    if not isinstance(weights, dict):
        raise ValueError("infer_json['weights'] must be dict")

    w = {}
    for a in selected:
        v = _safe_float(weights.get(a, 0.0), 0.0)
        if v < 0 or not np.isfinite(v):
            v = 0.0
        w[a] = float(v)

    s = sum(w.values())
    if s <= 0:
        w = {a: 1.0 / len(selected) for a in selected}
    else:
        w = {a: float(v / s) for a, v in w.items()}

    return selected, w, obj


def build_prior_equal(selected: List[str]) -> Dict[str, float]:
    k = len(selected)
    if k <= 0:
        return {}
    return {a: 1.0 / k for a in selected}


def build_prior_metrics(
    selected: List[str],
    metrics_csv: str,
    metric_col: str = "sharpe_net",
    temperature: float = 1.0,
) -> Dict[str, float]:
    # fallback equal if anything fails
    prior_eq = build_prior_equal(selected)
    if not metrics_csv or (not os.path.exists(metrics_csv)):
        return prior_eq

    try:
        m = pd.read_csv(metrics_csv)
    except Exception:
        return prior_eq

    if "alpha" not in m.columns or metric_col not in m.columns:
        return prior_eq

    m = m.copy()
    m["alpha"] = m["alpha"].astype(str)
    m[metric_col] = pd.to_numeric(m[metric_col], errors="coerce")

    score_map = m.set_index("alpha")[metric_col].to_dict()

    scores = []
    keys = []
    for a in selected:
        v = score_map.get(a, np.nan)
        v = float(v) if v is not None else np.nan
        if not np.isfinite(v):
            v = 0.0
        keys.append(a)
        scores.append(v)

    scores = np.array(scores, dtype=float)
    # softmax with temperature
    t = float(temperature) if temperature and temperature > 1e-9 else 1.0
    z = scores / t
    z = z - np.max(z)  # stability
    expz = np.exp(z)
    s = float(np.sum(expz))
    if s <= 0 or not np.isfinite(s):
        return prior_eq
    w = expz / s
    return {k: float(wi) for k, wi in zip(keys, w)}


def bayes_shrinkage(
    w_llm: Dict[str, float],
    w_prior: Dict[str, float],
    confidence: float,
) -> Dict[str, float]:
    c = float(confidence)
    c = min(max(c, 0.0), 1.0)

    keys = list(w_llm.keys())
    post = {}
    for k in keys:
        post[k] = c * float(w_llm.get(k, 0.0)) + (1.0 - c) * float(w_prior.get(k, 0.0))

    # non-negative + renorm
    for k in keys:
        if post[k] < 0 or not np.isfinite(post[k]):
            post[k] = 0.0

    s = sum(post.values())
    if s <= 0:
        # fallback equal
        k = len(keys)
        return {a: 1.0 / k for a in keys} if k > 0 else {}
    return {a: float(v / s) for a, v in post.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_json", type=str, required=True)
    ap.add_argument("--metrics_csv", type=str, default=None)

    ap.add_argument("--prior", type=str, default="metrics",
                    choices=["equal", "metrics"],
                    help="prior type: equal or metrics(softmax by metric_col)")
    ap.add_argument("--metric_col", type=str, default="sharpe_net",
                    help="metric column used for metrics prior (e.g. sharpe_net)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax temperature for metrics prior (higher -> closer to equal)")

    ap.add_argument("--confidence", type=float, default=0.6,
                    help="0-1 confidence on LLM weights. 1=use LLM, 0=use prior")

    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    selected, w_llm, raw = load_infer_json(args.infer_json)

    if args.prior == "equal":
        w_prior = build_prior_equal(selected)
    else:
        w_prior = build_prior_metrics(
            selected=selected,
            metrics_csv=args.metrics_csv or "",
            metric_col=args.metric_col,
            temperature=args.temperature,
        )

    w_post = bayes_shrinkage(w_llm=w_llm, w_prior=w_prior, confidence=args.confidence)

    out = {
        "selected": selected,
        "weights": w_post,
        "constraints": raw.get("constraints", {}),
        "notes": [
            f"bayes_shrinkage: prior={args.prior}, confidence={float(args.confidence):.3f}, metric_col={args.metric_col}, temp={float(args.temperature):.3f}",
            f"source_infer_json: {args.infer_json}",
        ]
    }

    out_path = args.out_json
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote bayes json: {out_path}")
    print("[INFO] Preview:")
    print(json.dumps(out, indent=2, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()