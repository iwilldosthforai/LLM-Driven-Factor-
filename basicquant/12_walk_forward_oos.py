import os
import json
import numpy as np
import pandas as pd

PANEL_PATH   = "data/processed/panel.parquet"
SIGNALS_PATH = "data/processed/signals.parquet"
OUT_NAV      = "results/walkforward_oos_nav.csv"
OUT_LOG      = "results/walkforward_oos_selected.json"

REB_FREQ = "W-FRI"
TOP_Q = 0.2
COST_BPS = 10

# walk-forward 设置
TRAIN_YEARS = 5
TEST_YEARS  = 1
K = 8                    # 每个窗口选 K 个因子
MAX_ABS_CORR = 0.8       # 训练窗口内的去相关阈值
BETA_LOOKBACK_WEEKS = 104

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
    var_m = mkt.rolling(lookback).var()
    cov = week_ret.rolling(lookback).cov(mkt)
    betas = cov.div(var_m, axis=0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return betas

def project_neutral(w: pd.Series, b: pd.Series) -> pd.Series:
    N = len(w)
    a1 = np.ones(N)
    a2 = b.values
    A = np.vstack([a1, a2])  # 2xN
    Aw = A @ w.values
    M = A @ A.T
    det = np.linalg.det(M)
    if abs(det) < 1e-12:
        return w - w.mean()
    Minv = np.linalg.inv(M)
    corr = A.T @ (Minv @ Aw)
    return pd.Series(w.values - corr, index=w.index)

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

    # 整体缩放：sum(w)=0 + L1=2
    def scale_grp(g):
        w = g["w"].to_numpy(dtype=float)
        w = w - w.mean()
        l1 = np.abs(w).sum()
        if l1 > 1e-12:
            w = w * (2.0 / l1)
        g["w"] = w
        return g

    df = df.groupby("reb_date", group_keys=False).apply(scale_grp)
    w = df.pivot(index="reb_date", columns="ticker", values="w").fillna(0.0)
    w.index.name = "date"
    return w

def compute_rankic_daily(sig_df: pd.DataFrame, fwd_df: pd.DataFrame) -> pd.Series:
    """sig_df: columns date,ticker,signal ; fwd_df: date,ticker,fwd_ret_5d"""
    df = sig_df.merge(fwd_df, on=["date","ticker"], how="left")
    def one_day(g):
        s = g["signal"]; y = g["fwd_ret_5d"]
        v = s.notna() & y.notna()
        if v.sum() < 10: return np.nan
        return s[v].rank().corr(y[v].rank())
    out = df.groupby("date", sort=True).apply(one_day)
    out.index = pd.to_datetime(out.index)
    return out

def greedy_select(alphas, score, corr_df, k, max_abs_corr):
    order = sorted(alphas, key=lambda a: score[a], reverse=True)
    sel = []
    for a in order:
        ok = True
        for s in sel:
            if abs(corr_df.loc[a, s]) > max_abs_corr:
                ok = False
                break
        if ok:
            sel.append(a)
        if len(sel) >= k:
            break
    return sel

def main():
    os.makedirs("results", exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date","ticker"])

    # 日收益 -> 周收益
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0

    # label for RankIC
    fwd = panel[["date","ticker","fwd_ret_5d"]].copy()

    sig = pd.read_parquet(SIGNALS_PATH)
    sig["date"] = pd.to_datetime(sig["date"])

    # 以“年”为窗口划分（用周频日期）
    years = sorted(list(set(week_ret.index.year)))
    # 从第 TRAIN_YEARS 年开始滚动
    logs = []
    oos_parts = []

    for start_i in range(0, len(years) - TRAIN_YEARS - TEST_YEARS + 1):
        train_years = years[start_i : start_i + TRAIN_YEARS]
        test_years  = years[start_i + TRAIN_YEARS : start_i + TRAIN_YEARS + TEST_YEARS]

        train_start = pd.Timestamp(f"{train_years[0]}-01-01")
        train_end   = pd.Timestamp(f"{train_years[-1]}-12-31")
        test_start  = pd.Timestamp(f"{test_years[0]}-01-01")
        test_end    = pd.Timestamp(f"{test_years[-1]}-12-31")

        # --- 训练窗口：算每个 alpha 的 RankIC 均值作为分数 ---
        train_sig = sig[(sig["date"]>=train_start) & (sig["date"]<=train_end)]
        alphas = sorted(train_sig["alpha"].unique().tolist())

        alpha_scores = {}
        alpha_weekly_ret = {}

        # 为相关性：先用训练窗口构建每个 alpha 的周收益序列（不做中性化也行）
        # 这里简单用：每个 alpha 单独跑一遍周权重 -> 周收益
        for a in alphas:
            s_a = train_sig[train_sig["alpha"]==a][["date","ticker","signal"]].rename(columns={"signal":"score"})
            w = build_weekly_weights(s_a, top_q=TOP_Q)
            # 对齐训练周收益
            wk_train = week_ret[(week_ret.index>=train_start) & (week_ret.index<=train_end)]
            common = w.index.intersection(wk_train.index)
            w = w.loc[common]
            wk = wk_train.loc[common, w.columns]
            r = (w * wk).sum(axis=1).fillna(0.0)
            alpha_weekly_ret[a] = r

            # RankIC 分数
            ric = compute_rankic_daily(
                train_sig[train_sig["alpha"]==a][["date","ticker","signal"]],
                fwd[(fwd["date"]>=train_start) & (fwd["date"]<=train_end)]
            )
            alpha_scores[a] = float(ric.mean(skipna=True))

        corr_df = pd.DataFrame(alpha_weekly_ret).corr().fillna(0.0)

        selected = greedy_select(alphas, alpha_scores, corr_df, K, MAX_ABS_CORR)

        logs.append({
            "train_years": train_years,
            "test_years": test_years,
            "selected": selected,
            "alpha_scores": {k: alpha_scores[k] for k in selected},
        })

        # --- 测试窗口：用 selected 做组合 + beta-neutral v2 ---
        test_sig = sig[(sig["date"]>=test_start) & (sig["date"]<=test_end)]
        test_score = test_sig[test_sig["alpha"].isin(selected)].groupby(["date","ticker"], as_index=False)["signal"].mean()
        test_score = test_score.rename(columns={"signal":"score"})

        w0 = build_weekly_weights(test_score, top_q=TOP_Q)

        wk_test = week_ret[(week_ret.index>=test_start) & (week_ret.index<=test_end)]
        common = w0.index.intersection(wk_test.index)
        w0 = w0.loc[common]
        wk = wk_test.loc[common, w0.columns]

        mkt = wk.mean(axis=1)
        betas = estimate_betas(wk, mkt, BETA_LOOKBACK_WEEKS)

        w_bn = w0.copy()
        for t in w_bn.index:
            b = betas.loc[t].reindex(w_bn.columns).fillna(0.0)
            w2 = project_neutral(w_bn.loc[t], b)
            l1 = w2.abs().sum()
            if l1 > 1e-12:
                w2 = w2 * (2.0 / l1)
            w_bn.loc[t] = w2

        gross = (w_bn * wk).sum(axis=1)
        to = 0.5 * w_bn.diff().abs().sum(axis=1).reindex(gross.index).fillna(0.0)
        net = gross - to * (COST_BPS/10000.0)

        part = pd.DataFrame({"date": net.index, "net_ret": net.values})
        part["train_start"] = train_start
        part["train_end"] = train_end
        part["test_start"] = test_start
        part["test_end"] = test_end
        oos_parts.append(part)

    oos = pd.concat(oos_parts, ignore_index=True).sort_values("date")
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    oos["nav"] = (1.0 + oos["net_ret"].fillna(0.0)).cumprod()

    oos.to_csv(OUT_NAV, index=False)
    with open(OUT_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    # 汇总 OOS 指标 + 市场 beta
    # 市场周收益（全样本）
    mkt_all = week_ret.mean(axis=1).reindex(pd.to_datetime(oos["date"])).dropna()
    strat = pd.Series(oos.set_index("date")["net_ret"]).reindex(mkt_all.index).dropna()
    beta, r2 = ols_beta(strat, mkt_all)

    print(f"Saved: {OUT_NAV}")
    print(f"Saved: {OUT_LOG}")
    print("Walk-forward OOS summary:")
    print("  weeks:", int(len(strat)))
    print("  sharpe:", sharpe(strat))
    print("  mdd:", max_drawdown(oos.set_index('date')["nav"]))
    print("  corr(strat,mkt):", float(strat.corr(mkt_all)))
    print("  beta:", beta, " R^2:", r2)

if __name__ == "__main__":
    main()