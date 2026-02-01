import os
import json
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# ====== Inputs ======
PANEL_PATH      = "data/processed/panel.parquet"
SIGNALS_PATH    = "data/processed/signals.parquet"          # seed signals
LLM_SIGNALS_DIR = "data/processed/signals_llm"              # optional partitioned parquet dataset

METRICS_CSV = "results/alpha_metrics.csv"                   # produced by 03_metrics_engine.py
CORR_NPY    = "results/alpha_corr.npy"                      # produced by 03_metrics_engine.py (weekly alpha return corr)
OUT_JSON    = "results/selected_alphas_v3.json"

# ====== Tunables ======
K = 8
CORR_MAX = 0.85

# 第一轮过滤阈值
MIN_WEEKS = 260
MIN_STAB  = 0.52
MAX_TURN  = 1.00
MIN_NET_SHARPE = 0.0

# 与 03/05 一致：周五权重 -> 下一周收益
TRADE_LAG_BDAYS = 0
REB_FREQ = "W-FRI"
TOP_Q = 0.2

# 成本鲁棒性：多个成本情景都要能活
COST_GRID_BPS = [0, 5, 10, 20]

# beta 惩罚（越大越偏向低 beta）
BETA_PENALTY = 0.30
MAX_ABS_BETA = None  # 可选硬过滤，如 0.6


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean()
    sd = s.std(ddof=1)
    if (sd is None) or (np.isnan(sd)) or (sd < 1e-12):
        return s * 0.0
    return (s - mu) / sd


def sharpe_weekly(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 30:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(mu / sd * np.sqrt(ann_factor))


def build_weekly_weights(sig: pd.Series, top_q=0.2, trade_lag_bdays: int = 0) -> pd.DataFrame:
    """
    周频多空等权权重（行和=0，多头和=+1，空头和=-1）
    ✅ 完全避免 groupby.apply，从根源避免 pandas 版本差异导致 reb_date 丢失
    """
    df = sig.rename("sig").reset_index()

    # 兼容：MultiIndex 没有 name 时 reset_index 会生成 level_0/level_1
    if "date" not in df.columns or "ticker" not in df.columns:
        ren = {}
        if "level_0" in df.columns: ren["level_0"] = "date"
        if "level_1" in df.columns: ren["level_1"] = "ticker"
        df = df.rename(columns=ren)

    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(f"sig.reset_index() must contain date/ticker columns. got columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["date", "ticker"])

    # 信号在 date 收盘后生成，最早在 date + trade_lag_bdays 个交易日收盘调仓生效
    if trade_lag_bdays and trade_lag_bdays > 0:
        df["trade_date"] = df["date"] + BDay(trade_lag_bdays)
    else:
        df["trade_date"] = df["date"]

    # 周五重平衡日
    df["reb_date"] = df["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

    # 每个 reb_date/ticker 取最后一个观测
    df = (
        df.sort_values(["reb_date", "ticker", "trade_date", "date"])
          .groupby(["reb_date", "ticker"], as_index=False)
          .tail(1)
    )

    # 截面分位排名
    df["rank"] = df.groupby("reb_date")["sig"].rank(method="average", pct=True)

    long_mask  = df["rank"] >= (1 - top_q)
    short_mask = df["rank"] <= top_q

    df["w"] = 0.0
    df.loc[long_mask,  "w"] =  1.0
    df.loc[short_mask, "w"] = -1.0

    # ✅ 向量化归一化（不会丢 reb_date）
    pos_n = df.groupby("reb_date")["w"].transform(lambda x: (x > 0).sum()).astype(float)
    neg_n = df.groupby("reb_date")["w"].transform(lambda x: (x < 0).sum()).astype(float)

    pos_n = pos_n.replace(0.0, np.nan)
    neg_n = neg_n.replace(0.0, np.nan)

    df.loc[df["w"] > 0, "w"] = df.loc[df["w"] > 0, "w"] / pos_n[df["w"] > 0]
    df.loc[df["w"] < 0, "w"] = df.loc[df["w"] < 0, "w"] / neg_n[df["w"] < 0]

    df["w"] = df["w"].fillna(0.0)

    weights = df.pivot(index="reb_date", columns="ticker", values="w").fillna(0.0)
    weights.index.name = "date"
    return weights


def compute_turnover(weights: pd.DataFrame) -> float:
    dw = weights.diff().abs()
    to = 0.5 * dw.sum(axis=1)
    to = to.replace([np.inf, -np.inf], np.nan).dropna()
    if len(to) == 0:
        return np.nan
    return float(to.mean())


def load_signals_concat() -> pd.DataFrame:
    sig_seed = pd.read_parquet(SIGNALS_PATH)

    sig_all = [sig_seed]
    if os.path.exists(LLM_SIGNALS_DIR):
        try:
            sig_llm = pd.read_parquet(LLM_SIGNALS_DIR)
            # 有些 parquet 会把 index 写进去
            if "date" not in sig_llm.columns or "ticker" not in sig_llm.columns:
                sig_llm = sig_llm.reset_index()
            if "date" not in sig_llm.columns and "level_0" in sig_llm.columns:
                sig_llm = sig_llm.rename(columns={"level_0": "date"})
            if "ticker" not in sig_llm.columns and "level_1" in sig_llm.columns:
                sig_llm = sig_llm.rename(columns={"level_1": "ticker"})
            sig_all.append(sig_llm)
        except Exception as e:
            print(f"[WARN] Failed to read {LLM_SIGNALS_DIR}: {e}. Continue with seed only.")

    sig = pd.concat(sig_all, ignore_index=True)

    # schema 统一
    for col in ["alpha", "ticker"]:
        if col in sig.columns:
            sig[col] = sig[col].astype(str)
    sig["date"] = pd.to_datetime(sig["date"])
    sig["signal"] = pd.to_numeric(sig["signal"], errors="coerce")

    need = {"date", "ticker", "alpha", "signal"}
    missing = need - set(sig.columns)
    if missing:
        raise ValueError(f"Signals missing columns: {missing}. got={list(sig.columns)}")

    sig = sig.drop_duplicates(subset=["date", "ticker", "alpha"], keep="last")
    return sig


def load_corr_matrix(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    读取 alpha_corr.npy 并对齐到 metrics 的 alpha 顺序。
    重要：corr 的顺序必须与生成它时的 alpha 顺序一致，否则去冗余会错位。
    这里用 metrics 的 alpha 作为“统一顺序”，并对缺失 alpha 做降级处理。
    """
    corr = np.load(CORR_NPY, allow_pickle=False)
    alpha_list = metrics_df["alpha"].astype(str).tolist()

    if corr.shape[0] != len(alpha_list) or corr.shape[1] != len(alpha_list):
        # 如果维度不匹配，说明 corr 是在不同 alpha 集合/顺序上生成的
        # 这里采取保守做法：构造一个“未知相关=0”的矩阵，并打印警告
        print(f"[WARN] alpha_corr.npy shape={corr.shape} != (N,N) with N={len(alpha_list)}. "
              f"Will fallback to identity corr (no de-corr filtering).")
        corr_df = pd.DataFrame(np.eye(len(alpha_list)), index=alpha_list, columns=alpha_list)
        return corr_df

    corr_df = pd.DataFrame(corr, index=alpha_list, columns=alpha_list)
    return corr_df


def main():
    # 0) 读 metrics：初筛依据 + corr 对齐依据
    m = pd.read_csv(METRICS_CSV)
    m["alpha"] = m["alpha"].astype(str)

    # 兜底：signed 指标如果不存在，就用 abs（因为 03 里可能没有输出 signed 列）
    if "mean_rankIC_1w_signed" not in m.columns:
        m["mean_rankIC_1w_signed"] = m["mean_rankIC_1w"].abs()
    if "ICIR_1w_signed" not in m.columns:
        m["ICIR_1w_signed"] = m["ICIR_1w"].abs()

    # 1) 第一轮过滤：保证“样本足够 + 稳定 + 换手不过高 + 净夏普不为负”
    cand = m[
        (m["n_weeks"] >= MIN_WEEKS) &
        (m["stability_pos_ratio_1w"] >= MIN_STAB) &
        (m["turnover_weekly"] <= MAX_TURN) &
        (m["sharpe_weekly_net"] > MIN_NET_SHARPE)
    ].copy()

    print(f"[INFO] metrics total={len(m)} after filter candidates={len(cand)}")
    if len(cand) == 0:
        raise RuntimeError("No candidates left after filtering. Relax thresholds.")

    # 2) 市场周收益（与 backtest 对齐：周五 -> 下一周收益）
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    panel = panel.sort_values(["date", "ticker"])

    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0
    week_ret_fwd = week_ret.shift(-1)

    mkt = week_ret_fwd.mean(axis=1).dropna()

    # 3) 只对候选 alpha 计算：beta/corr_mkt + 成本鲁棒 sharpe（用真实周收益序列）
    sig_all = load_signals_concat()

    cand_alphas = set(cand["alpha"].astype(str).tolist())
    sig_all = sig_all[sig_all["alpha"].astype(str).isin(cand_alphas)].copy()

    rows = []
    for alpha, g in sig_all.groupby("alpha"):
        sig_series = g.set_index(["date", "ticker"])["signal"].sort_index()
        w = build_weekly_weights(sig_series, top_q=TOP_Q, trade_lag_bdays=TRADE_LAG_BDAYS)

        common_dates = w.index.intersection(week_ret_fwd.index)
        if len(common_dates) < MIN_WEEKS:
            continue

        w = w.loc[common_dates]
        wk = week_ret_fwd.loc[common_dates, w.columns]

        r = (w * wk).sum(axis=1).dropna()
        if len(r) < MIN_WEEKS:
            continue

        to = compute_turnover(w)
        if np.isnan(to):
            continue

        rr = r.reindex(mkt.index).dropna()
        mm = mkt.reindex(rr.index).dropna()
        rr = rr.reindex(mm.index).dropna()
        mm = mm.reindex(rr.index).dropna()
        if len(rr) < MIN_WEEKS:
            continue

        var_m = float(mm.var(ddof=1))
        if var_m <= 1e-12:
            beta = np.nan
            corr_m = np.nan
        else:
            beta = float(np.cov(rr.to_numpy(), mm.to_numpy(), ddof=1)[0, 1] / var_m)
            corr_m = float(rr.corr(mm))

        sharpe_by_cost = {}
        for bps in COST_GRID_BPS:
            cost = to * (bps / 10000.0)
            r_net = rr - cost
            sharpe_by_cost[bps] = sharpe_weekly(r_net)

        robust_sharpe = float(np.nanmin(list(sharpe_by_cost.values())))

        rows.append({
            "alpha": str(alpha),
            "beta_mkt": beta,
            "corr_mkt": corr_m,
            "turnover_recalc": to,
            "robust_sharpe": robust_sharpe,
            **{f"sharpe_net_{bps}bps": sharpe_by_cost[bps] for bps in COST_GRID_BPS},
        })

    extra = pd.DataFrame(rows)
    print(f"[INFO] extra stats computed for {len(extra)} alphas")
    if len(extra) == 0:
        raise RuntimeError("No candidates produced extra stats (beta/robust sharpe). Check signals/panel alignment.")

    cand = cand.merge(extra, on="alpha", how="inner")
    if MAX_ABS_BETA is not None:
        cand = cand[cand["beta_mkt"].abs() <= float(MAX_ABS_BETA)].copy()
        print(f"[INFO] after beta hard filter: {len(cand)}")

    if len(cand) == 0:
        raise RuntimeError("No candidates left after beta filter. Relax thresholds.")

    # 4) 打分：robust_sharpe + icir + stab - turnover - mdd - beta penalty
    cand["z_robust_sharpe"] = zscore(cand["robust_sharpe"])
    cand["z_icir"]          = zscore(cand["ICIR_1w_signed"])
    cand["z_stab"]          = zscore(cand["stability_pos_ratio_1w"])
    cand["z_turn"]          = zscore(cand["turnover_weekly"])
    cand["z_mdd"]           = zscore(cand["mdd_net"].abs())
    cand["abs_beta"]        = cand["beta_mkt"].abs()

    cand["score"] = (
        0.45 * cand["z_robust_sharpe"] +
        0.20 * cand["z_icir"] +
        0.20 * cand["z_stab"] -
        0.20 * cand["z_turn"] -
        0.10 * cand["z_mdd"] -
        BETA_PENALTY * cand["abs_beta"]
    )

    cand = cand.sort_values("score", ascending=False)

    # 5) corr 去冗余（必须严格对齐到 metrics 的 alpha 顺序）
    corr_df = load_corr_matrix(m)

    selected = []
    for a in cand["alpha"].astype(str).tolist():
        if len(selected) >= K:
            break

        if len(selected) == 0:
            selected.append(a)
            continue

        # 如果 corr 里没有（极少数：维度不匹配 fallback），就当相关为 0
        ok = True
        for b in selected:
            if (a in corr_df.index) and (b in corr_df.columns):
                if float(abs(corr_df.loc[a, b])) > CORR_MAX:
                    ok = False
                    break
        if ok:
            selected.append(a)

    out = {
        "version": "v3",
        "K": K,
        "CORR_MAX": CORR_MAX,
        "TRADE_LAG_BDAYS": TRADE_LAG_BDAYS,
        "COST_GRID_BPS": COST_GRID_BPS,
        "BETA_PENALTY": BETA_PENALTY,
        "MAX_ABS_BETA": MAX_ABS_BETA,
        "filters": {
            "MIN_WEEKS": MIN_WEEKS,
            "MIN_STAB": MIN_STAB,
            "MAX_TURN": MAX_TURN,
            "MIN_NET_SHARPE": MIN_NET_SHARPE
        },
        "selected": selected,
        "preview": cand.head(30)[[
            "alpha", "score", "robust_sharpe", "beta_mkt", "corr_mkt",
            "turnover_weekly", "ICIR_1w_signed", "stability_pos_ratio_1w", "mdd_net"
        ]].to_dict(orient="records")
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("Selected:", selected)
    print("Saved:", OUT_JSON)


if __name__ == "__main__":
    main()