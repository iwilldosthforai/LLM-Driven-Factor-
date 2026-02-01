import os
import re
import json
import random
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ====== Paths (use your BASE env vars) ======
BASE = os.environ.get("BASE", "/scratch/users/ntu/shuyan00/nscc_qwen")
OUT_JSONL = os.environ.get("OUT_JSONL", os.path.join(BASE, "results/llm_candidates.jsonl"))

MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", os.path.join(BASE, "hf/models/Qwen2.5-7B-Instruct"))

# How many candidates to generate
N = int(os.environ.get("N_CANDIDATES", "300"))

# Seed for reproducibility
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)


# ====== DSL spec (keep it consistent with your evaluator) ======
ALLOWED_FIELDS = ["open", "high", "low", "close", "volume", "ret1", "vwap"]
ALLOWED_FUNCS = [
    "ret", "rank", "ts_rank", "ts_mean", "ts_std", "delta", "zscore",
    "corr", "cov", "decay_linear", "clip", "abs", "sign", "log1p"
]
WINDOWS = [3, 5, 10, 14, 20, 30, 60, 90]


SYSTEM_RULES = """
You are a quant researcher. You must output ONLY a JSON object, no markdown, no extra text.
The JSON must have keys: alpha, expr, group, version, notes.

Rules:
- expr must be a DSL expression composed ONLY of allowed fields and functions.
- Do NOT use future data. NO shift(-1). NO fwd_ret. NO labels.
- Keep expr short (<= 120 chars).
- Use only these fields: open, high, low, close, volume, ret1, vwap
- Use only these funcs: ret, rank, ts_rank, ts_mean, ts_std, delta, zscore, corr, cov, decay_linear, clip, abs, sign, log1p
- Windows n must be one of: 3,5,10,14,20,30,60,90
- group must be one of: mom, rev, vol, liq
- alpha must follow: llm_<group>_<3-digit>
"""

# prompt templates by group
PROMPTS = {
    "mom":  "Generate a momentum-style alpha using only the DSL. Prefer trend-following.",
    "rev":  "Generate a reversal-style alpha using only the DSL. Prefer mean-reversion.",
    "vol":  "Generate a volatility / risk alpha using only the DSL. Prefer stability/vol signals.",
    "liq":  "Generate a liquidity/volume alpha using only the DSL. Prefer volume/turnover patterns.",
}

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict:
    """
    Extract the first {...} JSON object from model output.
    """
    m = JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in output")
    obj = json.loads(m.group(0))
    return obj


def _is_valid_expr(expr: str) -> bool:
    # quick checks (we still rely on downstream evaluator)
    if len(expr) > 120:
        return False
    bad = ["shift(-1", "fwd", "label", "future", "target", "y="]
    if any(b in expr for b in bad):
        return False
    # must contain only allowed tokens roughly
    # allow parentheses, commas, numbers, underscores
    tokens = re.findall(r"[A-Za-z_]+", expr)
    for t in tokens:
        if t in ALLOWED_FIELDS:
            continue
        if t in ALLOWED_FUNCS:
            continue
        # allow "llm" etc? no, expr shouldn't contain
        return False
    # windows must be from set if any numbers appear
    nums = re.findall(r"\b\d+\b", expr)
    for n in nums:
        if int(n) not in WINDOWS and int(n) not in [0, 1, -1]:
            return False
    return True


def _fix_alpha_name(alpha: str, group: str, idx: int) -> str:
    # force llm_<group>_<3-digit>
    return f"llm_{group}_{idx:03d}"


def build_messages(group: str, idx: int) -> List[Dict]:
    # Add some randomness to encourage diversity
    w1, w2 = random.sample(WINDOWS, 2)
    field = random.choice(ALLOWED_FIELDS)
    hint = f"Try using window={w1} or {w2}, field={field}."

    user = f"{PROMPTS[group]} {hint} Output JSON only."
    return [
        {"role": "system", "content": SYSTEM_RULES.strip()},
        {"role": "user", "content": user},
    ]


def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    print("Loading model:", MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    ok, fail = 0, 0
    seen_expr = set()

    with open(OUT_JSONL, "w") as f:
        idx_map = {"mom": 0, "rev": 0, "vol": 0, "liq": 0}

        # balanced generation
        groups = (["mom", "rev", "vol", "liq"] * ((N + 3) // 4))[:N]

        for i, g in enumerate(groups, start=1):
            idx_map[g] += 1
            alpha_name = _fix_alpha_name("", g, idx_map[g])

            messages = build_messages(g, idx_map[g])
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            text = tok.decode(out[0], skip_special_tokens=True)

            try:
                obj = _extract_json(text)
                obj["group"] = obj.get("group", g)
                obj["version"] = obj.get("version", "v1")
                obj["alpha"] = alpha_name

                expr = obj.get("expr", "")
                if not isinstance(expr, str) or not expr:
                    raise ValueError("Missing expr")
                expr = expr.strip()

                if not _is_valid_expr(expr):
                    raise ValueError(f"Invalid expr: {expr}")

                if expr in seen_expr:
                    raise ValueError("Duplicate expr")
                seen_expr.add(expr)

                obj["expr"] = expr
                obj["notes"] = str(obj.get("notes", "")).strip()[:300]

                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                ok += 1

            except Exception as e:
                fail += 1
                # print a short debug line (kept in stdout log)
                print(f"[FAIL] i={i} group={g} err={e}")

    print(f"Saved: {OUT_JSONL}")
    print(f"Done. ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()