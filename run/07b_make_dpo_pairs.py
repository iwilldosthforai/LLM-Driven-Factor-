#!/usr/bin/env python3
# 07b_make_dpo_pairs.py
# Generate DPO preference pairs for portfolio JSON selection.
# Output: results/dpo_pairs.jsonl with {"prompt":..., "chosen":..., "rejected":...}

import os
import re
import json
import math
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ----------------------------
# utils
# ----------------------------
def print_gpu_info():
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[INFO] gpu_count: {n}")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"[INFO] gpu{i}: {name} cap={cap} vram={vram_gb:.1f}GB")


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    # ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # first balanced {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                s = text[start : i + 1]
                try:
                    return json.loads(s)
                except Exception:
                    return None
    return None


def repair_output(
    raw: Dict[str, Any],
    candidates: List[str],
    k: int,
    corr_max: float,
    turn_cap: float,
) -> Dict[str, Any]:
    cand_set = set(candidates)
    k = int(k)

    sel = raw.get("selected", [])
    if not isinstance(sel, list):
        sel = []
    sel_clean, seen = [], set()
    for a in sel:
        if isinstance(a, str) and a in cand_set and a not in seen:
            sel_clean.append(a)
            seen.add(a)

    # if len(candidates) == K, must select all in order
    if len(candidates) == k:
        sel_clean = list(candidates)
    else:
        if len(sel_clean) < k:
            for a in candidates:
                if a not in seen:
                    sel_clean.append(a)
                    seen.add(a)
                if len(sel_clean) >= k:
                    break
        sel_clean = sel_clean[:k]

    w_raw = raw.get("weights", {})
    if not isinstance(w_raw, dict):
        w_raw = {}

    weights = {}
    for a in sel_clean:
        if a in w_raw:
            v = safe_float(w_raw[a])
            if v is not None:
                weights[a] = v

    for a in sel_clean:
        if a not in weights:
            weights[a] = 1.0

    # no negatives
    for a in list(weights.keys()):
        if weights[a] < 0:
            weights[a] = 0.0

    s = sum(weights.values())
    if s <= 0:
        weights = {a: 1.0 / len(sel_clean) for a in sel_clean}
    else:
        weights = {a: float(v / s) for a, v in weights.items()}

    notes = raw.get("notes", [])
    if not isinstance(notes, list):
        notes = []
    notes = [str(x) for x in notes[:3]]

    out = {
        "selected": sel_clean,
        "weights": weights,
        "constraints": {"K": int(k), "CORR_MAX": float(corr_max), "TURN_CAP": float(turn_cap)},
        "notes": notes if notes else ["Auto-repaired to satisfy candidate-only + K/weights constraints."],
    }
    return out


def build_prompt(candidates: List[str], k: int, corr_max: float, turn_cap: float) -> str:
    cand_lines = "\n".join([f"- {c}" for c in candidates])
    if len(candidates) == int(k):
        selection_rule = "IMPORTANT: candidates count equals K, so you MUST select ALL candidates, no repeats, no extras."
    else:
        selection_rule = "Select exactly K unique alphas from candidates, no repeats, no extras."

    return f"""You are a quant portfolio assistant. Output MUST be a single valid JSON object, no extra text.

Task:
Given candidate alphas and constraints, return JSON with this exact schema keys:
{{
  "selected": ["alpha1","alpha2","alpha3","alpha4","alpha5"],
  "weights": {{"alpha1":0.22,"alpha2":0.18,"alpha3":0.25,"alpha4":0.20,"alpha5":0.15}},
  "constraints": {{"K": {k}, "CORR_MAX": {corr_max}, "TURN_CAP": {turn_cap}}},
  "notes": ["short justification ..."]
}}

Rules:
- {selection_rule}
- selected MUST be a subset of candidates.
- weights keys MUST match selected exactly.
- weights MUST be non-negative and sum to 1.0 (normalize if needed).
- Prefer higher net sharpe, lower beta, diversified exposures.
- Keep pairwise correlation under CORR_MAX as much as possible and respect turnover cap.

Candidates:
{cand_lines}

Constraints:
- K = {k}
- CORR_MAX = {corr_max}
- TURN_CAP = {turn_cap}

Now output the JSON only.
"""


# ----------------------------
# scoring (metrics + corr + turnover penalty)
# ----------------------------
def load_metrics(metrics_csv: str) -> pd.DataFrame:
    m = pd.read_csv(metrics_csv)
    if "alpha" not in m.columns:
        raise ValueError(f"metrics_csv missing 'alpha' column: {metrics_csv}")
    m["alpha"] = m["alpha"].astype(str)
    return m


def pick_metric_col(m: pd.DataFrame, candidates: List[str], prefer: List[str]) -> Optional[str]:
    for c in prefer:
        if c in m.columns:
            return c
    return None


def load_corr(corr_npy: str, corr_index_json: str) -> Tuple[np.ndarray, Dict[str, int]]:
    corr = np.load(corr_npy)
    with open(corr_index_json, "r") as f:
        idx = json.load(f)
    # idx can be {alpha: index} or {"alpha": idx}
    if not isinstance(idx, dict):
        raise ValueError("alpha_corr_index.json must be a dict")
    # ensure int mapping
    mapping = {}
    for k, v in idx.items():
        try:
            mapping[str(k)] = int(v)
        except Exception:
            pass
    return corr, mapping


def max_pair_corr(selected: List[str], corr: np.ndarray, idx: Dict[str, int]) -> float:
    pos = []
    for a in selected:
        if a in idx:
            pos.append(idx[a])
        else:
            # missing corr index => treat as very bad
            return 1.0
    if len(pos) < 2:
        return 0.0
    sub = corr[np.ix_(pos, pos)]
    # absolute off-diagonal max
    sub = np.abs(sub)
    np.fill_diagonal(sub, 0.0)
    return float(np.max(sub))


def score_portfolio(
    out_json: Dict[str, Any],
    metrics: pd.DataFrame,
    corr: np.ndarray,
    corr_idx: Dict[str, int],
    corr_max: float,
    turn_cap: float,
    sharpe_col: str,
    turnover_col: Optional[str],
    lam_corr: float = 10.0,
    lam_turn: float = 5.0,
) -> float:
    sel = out_json.get("selected", [])
    w = out_json.get("weights", {})
    if not isinstance(sel, list) or not isinstance(w, dict) or len(sel) == 0:
        return -1e9

    # metric lookup
    mm = metrics.set_index("alpha")
    base = 0.0
    turn = 0.0

    for a in sel:
        wi = safe_float(w.get(a, 0.0)) or 0.0
        if a in mm.index:
            si = safe_float(mm.loc[a, sharpe_col]) if sharpe_col in mm.columns else None
            si = si if si is not None else 0.0
            base += wi * float(si)

            if turnover_col and turnover_col in mm.columns:
                ti = safe_float(mm.loc[a, turnover_col])
                ti = ti if ti is not None else 0.0
                turn += wi * float(ti)
        else:
            # unknown alpha => penalize
            base -= 0.2 * wi

    maxc = max_pair_corr(sel, corr, corr_idx)
    pen_corr = max(0.0, maxc - float(corr_max))
    pen_turn = max(0.0, float(turn) - float(turn_cap))

    return float(base - lam_corr * pen_corr - lam_turn * pen_turn)


# ----------------------------
# model generate
# ----------------------------
def load_model(base_model_dir: str, adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


@torch.no_grad()
def sample_once(tok, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed: int) -> str:
    # per-sample seed
    g = torch.Generator(device=model.device) if torch.cuda.is_available() else None
    if g is not None:
        g.manual_seed(seed)

    inputs = tok(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        generator=g,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text


def read_alpha_list_from_jsonl(jsonl_path: str) -> List[str]:
    alphas = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                a = obj.get("alpha", None)
                if isinstance(a, str):
                    alphas.append(a)
            except Exception:
                continue
    # unique keep order
    seen = set()
    out = []
    for a in alphas:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)

    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--sft_adapter_dir", type=str, required=True)

    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--corr_npy", type=str, required=True)
    ap.add_argument("--corr_index_json", type=str, required=True)

    ap.add_argument("--candidates_jsonl", type=str, required=True, help="llm_candidates.jsonl that contains alpha names")

    ap.add_argument("--out_jsonl", type=str, default="results/dpo_pairs.jsonl")

    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--num_samples_per_prompt", type=int, default=12)

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--corr_max", type=float, default=0.85)
    ap.add_argument("--turn_cap", type=float, default=1.0)

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    proj = args.proj_root
    out_path = os.path.join(proj, args.out_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print_gpu_info()
    print(f"[INFO] base_model_dir: {args.base_model_dir}")
    print(f"[INFO] sft_adapter_dir: {args.sft_adapter_dir}")

    metrics_csv = os.path.join(proj, args.metrics_csv) if not os.path.isabs(args.metrics_csv) else args.metrics_csv
    corr_npy = os.path.join(proj, args.corr_npy) if not os.path.isabs(args.corr_npy) else args.corr_npy
    corr_index_json = os.path.join(proj, args.corr_index_json) if not os.path.isabs(args.corr_index_json) else args.corr_index_json
    candidates_jsonl = os.path.join(proj, args.candidates_jsonl) if not os.path.isabs(args.candidates_jsonl) else args.candidates_jsonl

    m = load_metrics(metrics_csv)
    sharpe_col = pick_metric_col(m, [], ["sharpe_net", "sharpe", "ic_mean", "ret_mean"])
    if sharpe_col is None:
        raise ValueError("metrics_csv missing sharpe-like column (sharpe_net/sharpe/ic_mean/ret_mean)")
    turnover_col = pick_metric_col(m, [], ["turnover", "avg_turnover", "turn", "to_mean"])
    print(f"[INFO] metrics sharpe_col={sharpe_col}, turnover_col={turnover_col}")

    corr, corr_idx = load_corr(corr_npy, corr_index_json)
    alpha_pool = read_alpha_list_from_jsonl(candidates_jsonl)

    if len(alpha_pool) < max(50, args.k * 10):
        # fallback: use metrics alpha list
        alpha_pool = m["alpha"].astype(str).tolist()
        print("[WARN] candidates_jsonl too small; fallback to metrics alpha list")

    # filter pool to those with corr+metrics (best effort)
    pool2 = []
    mm_set = set(m["alpha"].astype(str).tolist())
    for a in alpha_pool:
        if a in mm_set and a in corr_idx:
            pool2.append(a)
    if len(pool2) >= args.k * 20:
        alpha_pool = pool2
        print(f"[INFO] alpha_pool filtered to metrics+corr: {len(alpha_pool)}")

    tok, model = load_model(args.base_model_dir, args.sft_adapter_dir)

    print(f"[INFO] Generating DPO pairs: prompts={args.num_prompts}, samples/prompt={args.num_samples_per_prompt}")
    ok_pairs, skipped = 0, 0

    with open(out_path, "w") as f:
        for pi in range(args.num_prompts):
            # sample candidate set
            cand = random.sample(alpha_pool, k=args.k)
            prompt = build_prompt(cand, args.k, args.corr_max, args.turn_cap)

            scored: List[Tuple[float, str]] = []
            for si in range(args.num_samples_per_prompt):
                seed = args.seed + pi * 1000 + si
                text = sample_once(
                    tok, model, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                )
                raw = extract_first_json_obj(text) or {}
                repaired = repair_output(raw, cand, args.k, args.corr_max, args.turn_cap)

                s = score_portfolio(
                    repaired, m, corr, corr_idx,
                    corr_max=args.corr_max,
                    turn_cap=args.turn_cap,
                    sharpe_col=sharpe_col,
                    turnover_col=turnover_col,
                )
                scored.append((s, json.dumps(repaired, ensure_ascii=False)))

            scored.sort(key=lambda x: x[0], reverse=True)
            if len(scored) < 2:
                skipped += 1
                continue

            best_s, best_json = scored[0]
            worst_s, worst_json = scored[-1]

            # if too similar, skip
            if best_json == worst_json or (best_s - worst_s) < 1e-6:
                skipped += 1
                continue

            rec = {"prompt": prompt, "chosen": best_json, "rejected": worst_json}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok_pairs += 1

            if (pi + 1) % 10 == 0:
                print(f"[INFO] progress: {pi+1}/{args.num_prompts}, ok_pairs={ok_pairs}, skipped={skipped}")

    print(f"[OK] Saved DPO pairs: {out_path}")
    print(f"[INFO] ok_pairs={ok_pairs}, skipped={skipped}")


if __name__ == "__main__":
    main()