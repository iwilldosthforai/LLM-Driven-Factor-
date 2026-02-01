import os
import json
import numpy as np
import pandas as pd

PANEL_PATH   = "data/processed/panel.parquet"
SIGNALS_PATH = "data/processed/signals.parquet"
OUT_NAV      = "results/walkforward_oos_nav_dynamic.csv"
OUT_LOG      = "results/walkforward_oos_selected_dynamic.json"

REB_FREQ = "W-FRI"
TOP_Q = 0.2
COST_BPS = 10

TRAIN_YEARS = 5
TEST_YEARS  = 1
K = 8
MAX_ABS_CORR = 0.8
BETA_LOOKBACK_WEEKS = 104

# 动态加权窗口（周）
ALPHA_WEIGHT_LOOKBACK = 26
# 只给正表现因子权重（避免把错的因子越加越大）
USE_POSITIVE_ONLY = True

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
    A = np.vstack([np.ones(N), b.values])  # 2xN
    Aw = A @ w.values
    M = A @ A.T
    if abs(np.linalg.det(M)) < 1e-12:
        return w - w.mean()
    corr = A.T @ (np.linalg.inv(M) @ Aw)
    return pd.Series(w.values - corr, index=w.index)

def build_weekly_weights_from_score(score_long: pd.DataFrame, top_q=0.2) -> pd.DataFrame:
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

def compute_rankic_daily(sig_df: pd.DataFrame, fwd_df: pd.DataFrame) -> float:
    df = sig_df.merge(fwd_df, on=["date","ticker"], how="left")
    def one_day(g):
        s = g["signal"]; y = g["fwd_ret_5d"]
        v = s.notna() & y.notna()
        if v.sum() < 10: return np.nan
        return s[v].rank().corr(y[v].rank())
    ts = df.groupby("date", sort=True).apply(one_day)
    return float(ts.mean(skipna=True))

def greedy_select(alphas, score, corr_df, k, max_abs_corr):
    order = sorted(alphas, key=lambda a: score[a], reverse=True)
    sel = []
    for a in order:
        ok = True
        for s in sel:
            if abs(corr_df.loc[a, s]) > max_abs_corr:
                ok = False
                break
        if ok: sel.append(a)
        if len(sel) >= k: break
    return sel

def main():
    os.makedirs("results", exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date","ticker"])

    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0

    fwd = panel[["date","ticker","fwd_ret_5d"]].copy()

    sig = pd.read_parquet(SIGNALS_PATH)
    sig["date"] = pd.to_datetime(sig["date"])

    years = sorted(list(set(week_ret.index.year)))
    logs = []
    oos_parts = []

    for start_i in range(0, len(years) - TRAIN_YEARS - TEST_YEARS + 1):
        train_years = years[start_i : start_i + TRAIN_YEARS]
        test_years  = years[start_i + TRAIN_YEARS : start_i + TRAIN_YEARS + TEST_YEARS]

        train_start = pd.Timestamp(f"{train_years[0]}-01-01")
        train_end   = pd.Timestamp(f"{train_years[-1]}-12-31")
        test_start  = pd.Timestamp(f"{test_years[0]}-01-01")
        test_end    = pd.Timestamp(f"{test_years[-1]}-12-31")

        train_sig = sig[(sig["date"]>=train_start) & (sig["date"]<=train_end)]
        alphas = sorted(train_sig["alpha"].unique().tolist())

        alpha_scores = {}
        alpha_weekly_ret = {}

        wk_train = week_ret[(week_ret.index>=train_start) & (week_ret.index<=train_end)]

        # 用训练期周收益来做相关性 + 用训练期 rankIC 做分数
        for a in alphas:
            s_a = train_sig[train_sig["alpha"]==a][["date","ticker","signal"]].rename(columns={"signal":"score"})
            w = build_weekly_weights_from_score(s_a, top_q=TOP_Q)
            common = w.index.intersection(wk_train.index)
            w = w.loc[common]
            wk = wk_train.loc[common, w.columns]
            r = (w * wk).sum(axis=1).fillna(0.0)
            alpha_weekly_ret[a] = r

            alpha_scores[a] = compute_rankic_daily(
                train_sig[train_sig["alpha"]==a][["date","ticker","signal"]],
                fwd[(fwd["date"]>=train_start) & (fwd["date"]<=train_end)]
            )

        corr_df = pd.DataFrame(alpha_weekly_ret).corr().fillna(0.0)
        selected = greedy_select(alphas, alpha_scores, corr_df, K, MAX_ABS_CORR)

        logs.append({
            "train_years": train_years,
            "test_years": test_years,
            "selected": selected,
            "alpha_scores": {k: alpha_scores[k] for k in selected},
        })

        # ---- 测试期：动态加权组合信号（稳健版：直接 weekly 合成） ----
        test_sig = sig[(sig["date"]>=test_start) & (sig["date"]<=test_end)]
        wk_test = week_ret[(week_ret.index>=test_start) & (week_ret.index<=test_end)]

        # 1) 每个 alpha 生成 weekly (reb_date, ticker, sig)
        alpha_week_sig = {}
        alpha_test_weekret = {}

        for a in selected:
            s = test_sig[test_sig["alpha"]==a][["date","ticker","signal"]].copy()
            s["reb_date"] = s["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
            s = s.sort_values(["reb_date","ticker","date"]).groupby(["reb_date","ticker"]).tail(1)
            s = s[["reb_date","ticker","signal"]].rename(columns={"signal": a})
            alpha_week_sig[a] = s

            # 同时算该 alpha 的单因子周收益（用于rolling权重）
            s_a = test_sig[test_sig["alpha"]==a][["date","ticker","signal"]].rename(columns={"signal":"score"})
            w = build_weekly_weights_from_score(s_a, top_q=TOP_Q)
            common = w.index.intersection(wk_test.index)
            w = w.loc[common]
            wk = wk_test.loc[common, w.columns]
            r = (w * wk).sum(axis=1).fillna(0.0)
            alpha_test_weekret[a] = r

        alpha_test_weekret = pd.DataFrame(alpha_test_weekret).sort_index()

                # 2) rolling 计算每周的 alpha 权重 w_alpha[t]
        #    关键修复：权重只用到 t-1 的历史（避免 look-ahead）
        alpha_test_weekret = alpha_test_weekret.sort_index()

        # 滞后一周：t 的权重只能看到 <= t-1 的 alpha 周收益
        alpha_test_weekret_lag = alpha_test_weekret.shift(1)

        w_alpha = {}
        for t in alpha_test_weekret.index:
            # 取到 t 为止的“滞后收益”，由于 shift(1)，实际最后一行是 t-1 的收益
            hist = alpha_test_weekret_lag.loc[:t].tail(ALPHA_WEIGHT_LOOKBACK)

            # 再保险：如果 hist 最后一行全是 NaN（刚开始那几周），降级为等权
            if len(hist) < 8 or hist.dropna(how="all").shape[0] < 8:
                wa = pd.Series(1.0, index=selected, dtype=float)
            else:
                mu = hist.mean(skipna=True)
                sd = hist.std(ddof=1, skipna=True).replace(0.0, np.nan)

                # 你原来的 sharpe_like：均值/波动
                score = (mu / (sd + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # 只给“近期有效”的因子分配权重（可选）
                if USE_POSITIVE_ONLY:
                    score = score.clip(lower=0.0)

                # 防止全 0（例如全部都被 clip 掉）
                if float(score.abs().sum()) < 1e-12:
                    wa = pd.Series(1.0, index=selected, dtype=float)
                else:
                    wa = score.astype(float)

            # 归一化到和为 1
            wa = wa.reindex(selected).fillna(0.0)
            wa = wa / (float(wa.sum()) + 1e-12)

            w_alpha[t] = wa

        # 3) 把 weekly alpha signals 合成 weekly score_long：date(=reb_date), ticker, score
        # 先 merge 成一个大表 (reb_date,ticker, a1,a2,...)
        mats = []
        for a in selected:
            mats.append(alpha_week_sig[a])
        mat = mats[0]
        for i in range(1, len(mats)):
            mat = mat.merge(mats[i], on=["reb_date","ticker"], how="outer")
        mat = mat.fillna(0.0)

        # 对每个 reb_date 合成 score
        combined_rows = []
        for t, g in mat.groupby("reb_date"):
            if t not in w_alpha:
                continue
            wa = w_alpha[t].reindex(selected).fillna(0.0).values  # (K,)
            X = g[selected].values                                 # (n, K)
            sc = X @ wa
            combined_rows.append(pd.DataFrame({"date": t, "ticker": g["ticker"].values, "score": sc}))

        if len(combined_rows) == 0:
            # 这个测试窗口没有可用信号，跳过
            continue

        combined = pd.concat(combined_rows, ignore_index=True)

        # 4) 用 weekly combined score 构建权重
        w0 = build_weekly_weights_from_score(combined, top_q=TOP_Q)

        # beta-neutral
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

        # beta-neutral
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

    # 汇总
    strat = oos.set_index("date")["net_ret"]
    mkt_all = None  # 简化：不再重复算 beta，这里只给Sharpe/MDD
    print(f"Saved: {OUT_NAV}")
    print(f"Saved: {OUT_LOG}")
    print("Dynamic-weight Walk-forward OOS summary:")
    print("  weeks:", int(len(strat)))
    print("  sharpe:", sharpe(strat))
    print("  mdd:", max_drawdown(oos.set_index('date')["nav"]))

if __name__ == "__main__":
    main()