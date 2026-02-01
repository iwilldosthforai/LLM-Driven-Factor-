import os
import json
import numpy as np
import pandas as pd

PANEL_PATH   = "data/processed/panel.parquet"
SIGNALS_PATH = "data/processed/signals.parquet"
SELECT_PATH  = "results/selected_alphas.json"

OUT_NAV = "results/portfolio_nav_beta_neutral.csv"

TOP_Q = 0.2
COST_BPS = 10
REB_FREQ = "W-FRI"
BETA_LOOKBACK_WEEKS = 52  # 用过去一年估 beta

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

    def norm_grp(g):
        w = g["w"].to_numpy()
        pos = (w > 0).sum()
        neg = (w < 0).sum()
        if pos > 0:
            w[w > 0] = w[w > 0] / pos
        if neg > 0:
            w[w < 0] = w[w < 0] / neg
        g["w"] = w
        return g

    df = df.groupby("reb_date", group_keys=False).apply(norm_grp)
    w = df.pivot(index="reb_date", columns="ticker", values="w").fillna(0.0)
    w.index.name = "date"
    return w

def turnover(weights: pd.DataFrame) -> pd.Series:
    return 0.5 * weights.diff().abs().sum(axis=1)

def sharpe(x: pd.Series, ann=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(ann))

def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())

def ols_beta(y, x):
    x = x.dropna()
    y = y.reindex(x.index).dropna()
    x = x.reindex(y.index)
    if len(y) < 30:
        return np.nan, np.nan
    X = np.vstack([np.ones(len(x)), x.values]).T
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    resid = y.values - X @ b
    r2 = 1 - (resid.var() / (y.values.var() + 1e-12))
    return float(b[1]), float(r2)

def beta_neutralize_weights(weights: pd.DataFrame, week_ret: pd.DataFrame) -> pd.DataFrame:
    """
    对每周权重做 beta-neutral：
    - 估计每只股票 beta_i（用过去BETA_LOOKBACK_WEEKS周收益对市场收益回归）
    - 调整权重使 sum(w_i * beta_i)=0
    """
    # 市场周收益（等权）
    mkt = week_ret.mean(axis=1)

    w_adj = weights.copy()
    tickers = weights.columns

    for t in weights.index:
        # 用 t 之前的历史窗口估 beta
        hist_end = t - pd.Timedelta(days=1)
        hist = week_ret.loc[:hist_end].tail(BETA_LOOKBACK_WEEKS)
        if len(hist) < 30:
            continue

        m = mkt.loc[hist.index]
        # 逐 ticker 回归得到 beta
        betas = {}
        for col in tickers:
            y = hist[col]
            # beta = cov(y,m)/var(m)
            v = m.var()
            if v <= 1e-12:
                betas[col] = 0.0
            else:
                betas[col] = float(((y - y.mean()) * (m - m.mean())).mean() / v)

        beta_vec = pd.Series(betas).reindex(tickers).fillna(0.0)

        w = w_adj.loc[t].copy()
        # 如果该周持仓为空，跳过
        if w.abs().sum() <= 1e-12:
            continue

        # 目标：sum((w - λ*beta)*beta)=0 => λ = (w·beta)/(beta·beta)
        denom = float((beta_vec * beta_vec).sum())
        if denom <= 1e-12:
            continue
        lam = float((w * beta_vec).sum()) / denom
        w2 = w - lam * beta_vec

        # 重新归一化：多头和=+1，空头和=-1
        pos = w2[w2 > 0]
        neg = w2[w2 < 0]
        if len(pos) > 0:
            w2.loc[pos.index] = pos / pos.sum()
        if len(neg) > 0:
            w2.loc[neg.index] = neg / (-neg.sum())

        w_adj.loc[t] = w2

    return w_adj

def main():
    os.makedirs("results", exist_ok=True)

    with open(SELECT_PATH, "r") as f:
        sel = json.load(f)
    selected = [x["alpha"] for x in sel["selected"]]

    sig = pd.read_parquet(SIGNALS_PATH)
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig[sig["alpha"].isin(selected)]
    score = sig.groupby(["date","ticker"], as_index=False)["signal"].mean()
    score = score.rename(columns={"signal":"score"})

    weights = build_weekly_weights(score, top_q=TOP_Q)

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0

    common_dates = weights.index.intersection(week_ret.index)
    weights = weights.loc[common_dates]
    wk = week_ret.loc[common_dates, weights.columns]

    # beta neutralize
    weights_bn = beta_neutralize_weights(weights, wk)

    gross = (weights_bn * wk).sum(axis=1)
    to = turnover(weights_bn).reindex(gross.index).fillna(0.0)
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

    # 风险检查：对市场的 beta
    mkt = wk.mean(axis=1).reindex(net.index).dropna()
    beta, r2 = ols_beta(net, mkt)

    print(f"Saved: {OUT_NAV}")
    print("Beta-neutral summary:")
    print("  weeks:", int(len(net)))
    print("  sharpe_net:", sharpe(net))
    print("  mdd_net:", max_drawdown(nav))
    print("  avg_turnover:", float(to.mean()))
    print("  corr(strat, mkt):", float(net.corr(mkt)))
    print("  beta to mkt:", beta)
    print("  R^2:", r2)

if __name__ == "__main__":
    main()