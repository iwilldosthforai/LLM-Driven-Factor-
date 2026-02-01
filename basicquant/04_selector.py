import os
import json
import numpy as np
import pandas as pd

METRICS_PATH = "results/alpha_metrics.csv"
CORR_PATH    = "results/alpha_corr.npy"
OUT_JSON     = "results/selected_alphas.json"


#K = 10
#MIN_ICIR = 0.0
#MIN_STAB = 0.55
#MAX_TURNOVER = 1.2        
#MIN_MDD_NET = -0.99       
#MAX_ABS_CORR = 0.70       

K = 10
MIN_ICIR = -1e9
MIN_STAB = 0.50
MAX_TURNOVER = 10.0
MIN_MDD_NET = -0.95
MAX_ABS_CORR = 0.70

def main():
    os.makedirs("results", exist_ok=True)

    m = pd.read_csv(METRICS_PATH)
    corr = np.load(CORR_PATH)

    # corr.npy 只是矩阵，需要按 alpha 顺序对齐
    # 我们假设 corr 是按照 m 的 alpha 顺序保存的（03_metrics_engine.py 就是这样）
    alphas = m["alpha"].tolist()
    corr_df = pd.DataFrame(corr, index=alphas, columns=alphas)

    # 1) 硬过滤（把明显不可交易/不稳定的先踢掉）
    filt = (
        (m["ICIR_5d"] >= MIN_ICIR) &
        (m["stability_pos_ratio"] >= MIN_STAB) &
        (m["turnover_weekly"] <= MAX_TURNOVER) &
        (m["mdd_net"] >= MIN_MDD_NET) &
        (m["n_weeks"] >= 30)
    )
    cand = m.loc[filt].copy()

 
    if len(cand) < K:
        cand = m.loc[
            (m["ICIR_5d"] >= MIN_ICIR) &
            (m["stability_pos_ratio"] >= (MIN_STAB - 0.05)) &
            (m["mdd_net"] >= MIN_MDD_NET) &
            (m["n_weeks"] >= 30)
        ].copy()


    cand["score"] = (
    1.0 * cand["mean_rankIC_5d"] +
    0.4 * cand["ICIR_5d"] +
    0.3 * cand["stability_pos_ratio"] +
    0.3 * cand["sharpe_weekly_net"] -
    0.8 * cand["turnover_weekly"]
)

    cand = cand.sort_values("score", ascending=False)

    # 3) Greedy 去相关选 K
    selected = []
    for a in cand["alpha"]:
        ok = True
        for s in selected:
            if abs(corr_df.loc[a, s]) > MAX_ABS_CORR:
                ok = False
                break
        if ok:
            selected.append(a)
        if len(selected) >= K:
            break

    # 如果去相关导致选不够 K，就放宽相关阈值
    if len(selected) < K:
        for thr in [0.8, 0.9]:
            selected = []
            for a in cand["alpha"]:
                ok = True
                for s in selected:
                    if abs(corr_df.loc[a, s]) > thr:
                        ok = False
                        break
                if ok:
                    selected.append(a)
                if len(selected) >= K:
                    break
            if len(selected) >= K:
                break

    # 输出附带一些信息，便于审计
    out = {
        "K": K,
        "constraints": {
            "MIN_ICIR": MIN_ICIR,
            "MIN_STAB": MIN_STAB,
            "MAX_TURNOVER": MAX_TURNOVER,
            "MIN_MDD_NET": MIN_MDD_NET,
            "MAX_ABS_CORR": MAX_ABS_CORR,
        },
        "selected": [],
    }

    # 记录每个选中因子的指标摘要
    sub = m.set_index("alpha").loc[selected].reset_index()
    for _, r in sub.iterrows():
        out["selected"].append({
            "alpha": r["alpha"],
            "mean_rankIC_5d": float(r["mean_rankIC_5d"]),
            "ICIR_5d": float(r["ICIR_5d"]),
            "stability_pos_ratio": float(r["stability_pos_ratio"]),
            "turnover_weekly": float(r["turnover_weekly"]),
            "sharpe_weekly_net": float(r["sharpe_weekly_net"]),
            "mdd_net": float(r["mdd_net"]),
        })

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {OUT_JSON}")
    print("Selected alphas:")
    print(sub[["alpha","mean_rankIC_5d","ICIR_5d","stability_pos_ratio","turnover_weekly","sharpe_weekly_net","mdd_net"]])

if __name__ == "__main__":
    main()
