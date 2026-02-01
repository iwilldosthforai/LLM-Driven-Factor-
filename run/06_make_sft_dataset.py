import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / len(x)

def pick_metric_cols(metrics: pd.DataFrame) -> List[str]:
    # 按优先级挑可用列，作为 teacher 的打分依据
    candidates = [
        "robust_sharpe",
        "sharpe_weekly_net",
        "sharpe_weekly",
        "ICIR_1w",
        "mean_rankIC_1w",
        "ICIR_5d",
        "mean_rankIC_5d",
    ]
    return [c for c in candidates if c in metrics.columns]

def build_corr_df(corr_npy: str, alpha_index_json: str, metrics_alphas: List[str]) -> pd.DataFrame:
    """
    兼容两种情况：
    - 你已有 alpha_corr_index.json（推荐）
    - 没有 index json：则用 metrics 的 alpha 顺序（风险：需确保一致）
    """
    corr = np.load(corr_npy)
    if os.path.exists(alpha_index_json):
        with open(alpha_index_json, "r") as f:
            idx = json.load(f)
        # idx 可以是 list 或 dict
        if isinstance(idx, dict) and "alphas" in idx:
            alphas = [str(x) for x in idx["alphas"]]
        elif isinstance(idx, list):
            alphas = [str(x) for x in idx]
        else:
            raise ValueError(f"Unknown index json format: {alpha_index_json}")
    else:
        # 退化：用 metrics 里的 alpha 顺序
        alphas = [str(a) for a in metrics_alphas]

    if corr.shape[0] != len(alphas):
        raise ValueError(f"corr shape {corr.shape} not match index len {len(alphas)}")

    return pd.DataFrame(corr, index=alphas, columns=alphas)

def greedy_select_with_corr(pool: List[str], score_map: Dict[str, float], corr_df: pd.DataFrame, k: int, corr_max: float) -> List[str]:
    # pool 内按 score 降序
    pool_sorted = sorted(pool, key=lambda a: score_map.get(a, -1e9), reverse=True)
    selected = []
    for a in pool_sorted:
        if len(selected) >= k:
            break
        ok = True
        for b in selected:
            if a in corr_df.index and b in corr_df.columns:
                if float(abs(corr_df.loc[a, b])) > corr_max:
                    ok = False
                    break
        if ok:
            selected.append(a)
    return selected

def make_weights(selected: List[str], score_map: Dict[str, float], temperature: float = 0.7) -> Dict[str, float]:
    # 权重用 softmax(正分)；分数<=0 的也给一点点权重（避免全 0）
    scores = np.array([score_map.get(a, 0.0) for a in selected], dtype=float)
    # shift 到非负
    minv = np.min(scores)
    scores = scores - minv
    scores = np.clip(scores, 0.0, None)
    if np.all(scores <= 1e-12):
        w = np.ones(len(selected)) / len(selected)
    else:
        w = softmax(scores, temperature=temperature)
    return {a: float(w[i]) for i, a in enumerate(selected)}

def format_input_block(pool: List[str], m_sub: pd.DataFrame, corr_edges: List[Tuple[str,str,float]], k: int, corr_max: float, turn_cap: float) -> str:
    # 控制输入长度：只给 pool 的简表 + corr top edges（稀疏）
    cols = [c for c in ["alpha","mean_rankIC_1w","ICIR_1w","stability_pos_ratio_1w","turnover_weekly","sharpe_weekly_net","mdd_net"] if c in m_sub.columns]
    tbl = m_sub[cols].copy()
    tbl = tbl.head(min(len(tbl), 30))
    metrics_lines = tbl.to_csv(index=False)

    edges_lines = "\n".join([f"{a},{b},{v:.3f}" for a,b,v in corr_edges[:120]])

    text = f"""You are a quant research assistant. Given candidate alphas and their metrics, output a JSON with selected alphas and alpha-level weights.

Constraints:
- Select exactly K={k} alphas.
- Absolute pairwise correlation among selected alphas must be <= CORR_MAX={corr_max}.
- Prefer higher net Sharpe / robust Sharpe, higher ICIR, higher stability, lower turnover, lower drawdown.
- Turnover cap guideline: TURN_CAP={turn_cap} (weekly average).

Candidates (pool size={len(pool)}):
{",".join(pool)}

Metrics table (csv):
{metrics_lines}

Correlation edges (csv: alpha_i,alpha_j,corr):
{edges_lines}

Return ONLY a JSON object with keys: selected, weights, constraints, notes.
"""
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True, help="basicquant root (contains data/, results/)")
    ap.add_argument("--out_jsonl", type=str, default="results/sft_train.jsonl")
    ap.add_argument("--n_samples", type=int, default=1200)
    ap.add_argument("--pool_size", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--corr_max", type=float, default=0.85)
    ap.add_argument("--turn_cap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    proj = args.proj_root
    metrics_csv = os.path.join(proj, "results/alpha_metrics.csv")
    corr_npy    = os.path.join(proj, "results/alpha_corr.npy")
    corr_idx_js = os.path.join(proj, "results/alpha_corr_index.json")
    out_path    = os.path.join(proj, args.out_jsonl)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(metrics_csv)
    if not os.path.exists(corr_npy):
        raise FileNotFoundError(corr_npy)

    m = pd.read_csv(metrics_csv)
    m["alpha"] = m["alpha"].astype(str)

    metric_cols = pick_metric_cols(m)
    if len(metric_cols) == 0:
        raise RuntimeError(f"No usable metric cols in {metrics_csv}")

    # teacher score：优先 robust_sharpe，否则 sharpe_weekly_net，否则 ICIR_1w ...
    base_col = metric_cols[0]
    m["_teacher_score"] = pd.to_numeric(m[base_col], errors="coerce").fillna(0.0)

    # 稍微惩罚换手/回撤（如果有）
    if "turnover_weekly" in m.columns:
        m["_teacher_score"] = m["_teacher_score"] - 0.10 * pd.to_numeric(m["turnover_weekly"], errors="coerce").fillna(0.0)
    if "mdd_net" in m.columns:
        m["_teacher_score"] = m["_teacher_score"] - 0.05 * pd.to_numeric(m["mdd_net"], errors="coerce").fillna(0.0).abs()

    score_map = {str(r["alpha"]): float(r["_teacher_score"]) for _, r in m.iterrows()}

    corr_df = build_corr_df(corr_npy, corr_idx_js, m["alpha"].tolist())

    rng = np.random.default_rng(args.seed)
    alphas_all = m["alpha"].astype(str).unique().tolist()
    if len(alphas_all) < args.pool_size:
        raise RuntimeError(f"Not enough alphas: {len(alphas_all)} < pool_size {args.pool_size}")

    rows = []
    for i in range(args.n_samples):
        pool = rng.choice(alphas_all, size=args.pool_size, replace=False).tolist()
        m_sub = m[m["alpha"].isin(pool)].copy().sort_values("_teacher_score", ascending=False)

        # 生成稀疏 corr edges：每个 alpha 取 top-5 绝对相关
        corr_edges = []
        for a in pool:
            if a not in corr_df.index:
                continue
            s = corr_df.loc[a, pool].dropna().abs().sort_values(ascending=False)
            for b in s.index[:6]:
                if a == b:
                    continue
                corr_edges.append((a, str(b), float(corr_df.loc[a, b])))
        # 去重
        seen = set()
        uniq = []
        for a,b,v in corr_edges:
            key = tuple(sorted([a,b]))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((a,b,v))
        corr_edges = sorted(uniq, key=lambda x: abs(x[2]), reverse=True)

        selected = greedy_select_with_corr(pool, score_map, corr_df, k=args.k, corr_max=args.corr_max)
        # 如果因为 corr 太严选不满，放宽一点点（teacher 兜底）
        if len(selected) < args.k:
            selected = greedy_select_with_corr(pool, score_map, corr_df, k=args.k, corr_max=0.99)
        selected = selected[:args.k]
        if len(selected) < args.k:
            # 实在不够，随机补齐（极少数）
            rest = [a for a in pool if a not in selected]
            rng.shuffle(rest)
            selected = selected + rest[:(args.k-len(selected))]

        weights = make_weights(selected, score_map, temperature=0.7)

        x = format_input_block(pool, m_sub, corr_edges, args.k, args.corr_max, args.turn_cap)
        y = {
            "selected": selected,
            "weights": weights,
            "constraints": {"K": args.k, "CORR_MAX": args.corr_max, "TURN_CAP": args.turn_cap},
            "notes": [f"Teacher uses {base_col} with mild penalties on turnover/drawdown; greedy corr de-dup."]
        }

        rows.append({"input": x, "output": json.dumps(y, ensure_ascii=False)})

        if (i+1) % 200 == 0:
            print(f"[INFO] generated {i+1}/{args.n_samples}")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] saved: {out_path}  n={len(rows)}  base_col={base_col}")
    print("[TIP] next: run 07_sft_lora_train.py on GPU")

if __name__ == "__main__":
    main()