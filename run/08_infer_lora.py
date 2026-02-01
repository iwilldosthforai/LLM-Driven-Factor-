# 08_infer_lora_select.py
import os
import re
import json
import math
import argparse
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def print_env_and_gpu():
    print(f"[INFO] python: {os.popen('python -V').read().strip()}")
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
    else:
        print("[WARN] CUDA not available, will run on CPU (slow).")


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def load_candidates_from_select_json(path: str) -> List[str]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "selected" in obj:
        sel = obj["selected"]
        out = []
        for x in sel:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) and "alpha" in x:
                out.append(str(x["alpha"]))
        return [str(x) for x in out]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError(f"Unknown select_json format: {path}")


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试提取完整 JSON object（括号平衡）。
    """
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


def salvage_selected_weights(text: str) -> Dict[str, Any]:
    """
    关键抢救：即使 JSON 被 notes 截断，也尽量从文本中提取 selected + weights。
    """
    out: Dict[str, Any] = {}

    # selected: "selected": [ ... ]
    m_sel = re.search(r'"selected"\s*:\s*(\[[^\]]*\])', text, flags=re.S)
    if m_sel:
        try:
            out["selected"] = json.loads(m_sel.group(1))
        except Exception:
            pass

    # weights: "weights": { ... }
    # 用括号平衡抢救 dict
    m_w = re.search(r'"weights"\s*:\s*\{', text)
    if m_w:
        start = m_w.start()
        # 找到第一个 { 的位置
        brace0 = text.find("{", start)
        if brace0 >= 0:
            depth = 0
            for i in range(brace0, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        s = text[brace0 : i + 1]
                        try:
                            out["weights"] = json.loads(s)
                        except Exception:
                            pass
                        break

    return out


def repair_output(
    raw: Dict[str, Any],
    candidates: List[str],
    k: int,
    corr_max: float,
    turn_cap: float,
) -> Dict[str, Any]:
    cand = [str(x) for x in candidates]
    cand_set = set(cand)
    k = int(k)

    # 候选数==K -> 必须全选
    if len(cand) == k:
        selected = list(cand)
    else:
        sel = raw.get("selected", [])
        if not isinstance(sel, list):
            sel = []
        seen = set()
        selected = []
        for a in sel:
            if isinstance(a, str) and a in cand_set and a not in seen:
                selected.append(a)
                seen.add(a)
        if len(selected) < k:
            for a in cand:
                if a not in seen:
                    selected.append(a)
                    seen.add(a)
                if len(selected) >= k:
                    break
        selected = selected[:k]

    # weights
    w_raw = raw.get("weights", {})
    if not isinstance(w_raw, dict):
        w_raw = {}

    weights: Dict[str, float] = {}
    for a in selected:
        if a in w_raw:
            v = safe_float(w_raw[a])
            if v is not None:
                weights[a] = v

    for a in selected:
        if a not in weights:
            weights[a] = 1.0

    for a in list(weights.keys()):
        if weights[a] < 0:
            weights[a] = 0.0

    s = sum(weights.values())
    if s <= 0:
        weights = {a: 1.0 / len(selected) for a in selected}
    else:
        weights = {a: float(v / s) for a, v in weights.items()}

    notes = raw.get("notes", [])
    if not isinstance(notes, list):
        notes = []
    notes = [str(x) for x in notes[:1]]  # 强制只留 1 条
    if not notes:
        notes = ["Auto-repaired (salvaged selected/weights if output truncated)."]

    return {
        "selected": selected,
        "weights": weights,
        "constraints": {"K": int(k), "CORR_MAX": float(corr_max), "TURN_CAP": float(turn_cap)},
        "notes": notes,
    }


def build_messages(candidates: List[str], k: int, corr_max: float, turn_cap: float) -> List[Dict[str, str]]:
    cand_lines = "\n".join([f"- {c}" for c in candidates])

    if len(candidates) == int(k):
        rule = "IMPORTANT: candidates count equals K. You MUST select ALL candidates exactly once, no repeats, no extras."
    else:
        rule = "Select exactly K unique alphas from candidates (no repeats, no extras)."

    system = (
        "You are a quant portfolio assistant. "
        "Output ONLY one valid JSON object. No extra text. "
        "Never output 'Human:' or 'Assistant:'."
    )

    # 关键：notes 限制很短，避免被截断
    user = f"""
Return JSON with EXACT schema:
{{
  "selected": ["alpha1","alpha2","alpha3","alpha4","alpha5"],
  "weights": {{"alpha1":0.22,"alpha2":0.18,"alpha3":0.25,"alpha4":0.20,"alpha5":0.15}},
  "constraints": {{"K": {k}, "CORR_MAX": {corr_max}, "TURN_CAP": {turn_cap}}},
  "notes": ["ONE short sentence <= 16 words"]
}}

Rules:
- {rule}
- selected MUST be subset of candidates.
- weights keys MUST match selected exactly.
- weights MUST be non-negative and sum to 1.0.
- notes MUST be exactly 1 item and short (<=16 words).
- Output JSON ONLY.

Candidates:
{cand_lines}

Constraints:
K={k}, CORR_MAX={corr_max}, TURN_CAP={turn_cap}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)

    ap.add_argument("--candidates", type=str, nargs="*", default=None)
    ap.add_argument("--select_json", type=str, default=None)

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--corr_max", type=float, default=0.85)
    ap.add_argument("--turn_cap", type=float, default=1.0)

    ap.add_argument("--max_new_tokens", type=int, default=1024)  # ✅ 默认变大，避免 notes 截断
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="results/infer_lora_output.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print_env_and_gpu()
    print(f"[INFO] base_model_dir: {args.base_model_dir}")
    print(f"[INFO] adapter_dir: {args.adapter_dir}")

    # candidates
    if args.candidates and len(args.candidates) > 0:
        candidates = [str(x) for x in args.candidates]
    elif args.select_json:
        candidates = load_candidates_from_select_json(args.select_json)
    else:
        raise SystemExit("[FATAL] Provide either --candidates ... or --select_json path")

    # 去重保持顺序
    seen = set()
    cand2 = []
    for c in candidates:
        if c not in seen:
            cand2.append(c)
            seen.add(c)
    candidates = cand2

    # load
    tok = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = build_messages(candidates, args.k, args.corr_max, args.turn_cap)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    bad_words = ["Human:", "Assistant:"]
    bad_words_ids = [tok.encode(w, add_special_tokens=False) for w in bad_words if tok.encode(w, add_special_tokens=False)]

    print("[INFO] Run inference (deterministic, no sampling)...")
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    # 只解码生成部分
    gen_tokens = gen[0, input_len:]
    text = tok.decode(gen_tokens, skip_special_tokens=True).strip()

    print("\n========== MODEL OUTPUT (GENERATED ONLY) ==========\n")
    print(text)

    raw_json = extract_first_json_obj(text)
    if raw_json is None:
        # ✅ 抢救 selected/weights（notes 截断也不怕）
        raw_json = salvage_selected_weights(text)

    repaired = repair_output(
        raw=raw_json or {},
        candidates=candidates,
        k=args.k,
        corr_max=args.corr_max,
        turn_cap=args.turn_cap,
    )

    out_path = args.out_json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(repaired, f, indent=2, ensure_ascii=False)

    print("\n========== FINAL JSON (REPAIRED) ==========\n")
    print(json.dumps(repaired, indent=2, ensure_ascii=False))
    print(f"[INFO] saved to: {out_path}")


if __name__ == "__main__":
    main()