import os
import json
import numpy as np
import pandas as pd

PANEL_PATH   = "data/processed/panel.parquet"
SIGNALS_PATH = "data/processed/signals.parquet"
SELECT_PATH  = "results/selected_alphas.json"

OUT_NAV = "results/portfolio_nav_beta_neutral_v2.csv"

TOP_Q = 0.2
COST_BPS = 10
REB_FREQ = "W-FRI"
BETA_LOOKBACK_WEEKS = 104  # 拉长一点更稳（2年）

def build_weekly_weights(score_long: pd.DataFrame, top_q=0.2) -> pd.DataFrame:
    df = score_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","ticker"])
    df["reb_date"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    df = df.sort_values(["reb_date","ticker","date"]).groupby(["reb_date","ticker"]).tail(1)

    df["rank"] = df.groupby("reb_date")["score"].rank(method="average", pct=True)
    long_mask  = df["rank"] >= (1 - top_q)
    short_mask = df["rank"] <= top_q

    df["w"] = 0.0
    df.loc[long_mask, "w"] =  1.0
    df.loc[short_mask,"w"] = -1.0

    # 只做一次整体缩放（不要分正负归一化，否则会破坏中性化）
    def scale_grp(g):
        w = g["w"].to_numpy(dtype=float)
        # 资金中性：去均值（可选）
        w = w - w.mean()
        # L1=2（方便比较）：sum|w| = 2
        l1 = np.abs(w).sum()
        if l1 > 1e-12:
            w = w * (2.0 / l1)
        g["w"] = w
        return g

    df = df.groupby("reb_date", group_keys=False).apply(scale_grp)

    w = df.pivot(index="reb_date", columns="ticker", values="w").fillna(0.0)
    w.index.name = "date"
    return w

def turnover(weights: pd.DataFrame) -> pd.Series:
    return 0.5 * weights.diff().abs().sum(axis=1)

def sharpe(x: pd.Series, ann=52) -> float:
    x = x.dropna()
    if len(x) < 10: return float("nan")
    sd = x.std(ddof=1)
    if sd <= 1e-12: return float("nan")
    return float(x.mean() / sd * np.sqrt(ann))

def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())

def ols_beta(y, x):
    x = x.dropna()
    y = y.reindex(x.index).dropna()
    x = x.reindex(y.index)
    if len(y) < 30: return np.nan, np.nan
    X = np.vstack([np.ones(len(x)), x.values]).T
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    resid = y.values - X @ b
    r2 = 1 - (resid.var() / (y.values.var() + 1e-12))
    return float(b[1]), float(r2)

def estimate_betas(week_ret: pd.DataFrame, mkt: pd.Series, lookback: int) -> pd.DataFrame:
    """
    返回 betas[t, ticker]：在每个周t，用之前lookback周估计 beta_i = cov(r_i, mkt)/var(mkt)
    """
    betas = pd.DataFrame(index=week_ret.index, columns=week_ret.columns, dtype=float)
    var_m = mkt.rolling(lookback).var()
    cov = week_ret.rolling(lookback).cov(mkt)
    betas.loc[:] = cov.div(var_m, axis=0)
    betas = betas.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return betas

def project_neutral(w: pd.Series, b: pd.Series) -> pd.Series:
    """
    投影到约束空间：
    约束1：sum(w)=0
    约束2：w·b = 0
    目标：最小化 ||w* - w||^2
    解：w* = w - A^T (A A^T)^-1 (A w)
    其中 A 是 2 x N，第一行全1，第二行是 b
    """
    N = len(w)
    a1 = np.ones(N)
    a2 = b.values
    A = np.vstack([a1, a2])  # 2xN
    Aw = A @ w.values        # 2x1
    M = A @ A.T              # 2x2

    # 防止奇异
    det = np.linalg.det(M)
    if abs(det) < 1e-12:
        # 退化时只做 sum(w)=0
        w2 = w - w.mean()
        return w2

    Minv = np.linalg.inv(M)
    correction = A.T @ (Minv @ Aw)
    w2 = w.values - correction
    return pd.Series(w2, index=w.index)

def main():
    os.makedirs("results", exist_ok=True)

    with open(SELECT_PATH, "r") as f:
        sel = json.load(f)
    selected = [x["alpha"] for x in sel["selected"]]

    sig = pd.read_parquet(SIGNALS_PATH)
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig[sig["alpha"].isin(selected)]
    score = sig.groupby(["date","ticker"], as_index=False)["signal"].mean().rename(columns={"signal":"score"})

    # 原始周权重
    weights = build_weekly_weights(score, top_q=TOP_Q)

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0

    # 对齐
    common_dates = weights.index.intersection(week_ret.index)
    weights = weights.loc[common_dates]
    wk = week_ret.loc[common_dates, weights.columns]

    # 市场基准（等权）
    mkt = wk.mean(axis=1)

    # 估计每周每只股票 beta
    betas = estimate_betas(wk, mkt, BETA_LOOKBACK_WEEKS)

    # 对每周做投影中性化 + L1缩放
    w_bn = weights.copy()
    for t in w_bn.index:
        w = w_bn.loc[t]
        b = betas.loc[t].reindex(w.index).fillna(0.0)
        w2 = project_neutral(w, b)

        # 再做 L1 缩放保持规模（不破坏约束）
        l1 = w2.abs().sum()
        if l1 > 1e-12:
            w2 = w2 * (2.0 / l1)
        w_bn.loc[t] = w2

    gross = (w_bn * wk).sum(axis=1)
    to = turnover(w_bn).reindex(gross.index).fillna(0.0)
    net = gross - to * (COST_BPS/10000.0)
    nav = (1+net.fillna(0.0)).cumprod()

    out = pd.DataFrame({
        "date": gross.index,
        "gross_ret": gross.values,
        "net_ret": net.values,
        "turnover": to.values,
        "nav_net": nav.values,
    })
    out.to_csv(OUT_NAV, index=False)

    beta, r2 = ols_beta(net, mkt.reindex(net.index))
    print(f"Saved: {OUT_NAV}")
    print("Beta-neutral v2 summary:")
    print("  weeks:", int(len(net)))
    print("  sharpe_net:", sharpe(net))
    print("  mdd_net:", max_drawdown(nav))
    print("  avg_turnover:", float(to.mean()))
    print("  corr(strat, mkt):", float(net.corr(mkt.reindex(net.index))))
    print("  beta to mkt:", beta)
    print("  R^2:", r2)

if __name__ == "__main__":
    main()