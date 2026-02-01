import os
import json
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

PANEL_PATH   = "data/processed/panel.parquet"
SIGNALS_PATH = "data/processed/signals.parquet"
OUT_METRICS  = "results/alpha_metrics.csv"
OUT_CORR     = "results/alpha_corr.npy"
OUT_CORR_IDX = "results/alpha_corr_index.json"

# 默认设定
TOP_Q = 0.2            # long/short 分位数
COST_BPS = 10          # 10 bps 单边成本
REB_FREQ = "W-FRI"     # 周五重平衡

# 交易执行滞后（更保守可设为 1：信号滞后 1 个交易日才可调仓）
TRADE_LAG_BDAYS = 0


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    x = a.rank()
    y = b.rank()
    if x.notna().sum() < 10:
        return np.nan
    return x.corr(y)


def monthly_pos_ratio(rankic_ts: pd.Series) -> float:
    m = rankic_ts.resample("ME").mean().dropna()
    if len(m) == 0:
        return np.nan
    return float((m > 0).mean())


def build_weekly_weights(sig: pd.Series, top_q=0.2, trade_lag_bdays: int = 0) -> pd.DataFrame:
    """
    周频多空等权权重（行和=0，多头和=+1，空头和=-1）
    输入 sig: index=(date,ticker) 的 Series（信号越大越看多）
    输出 weights: index=reb_date, columns=ticker
    """
    df = sig.rename("sig").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["date", "ticker"])

    # 交易可用日期：信号在 date 收盘后生成，最早在 date + trade_lag_bdays 收盘调仓生效
    if trade_lag_bdays and trade_lag_bdays > 0:
        df["trade_date"] = df["date"] + BDay(trade_lag_bdays)
    else:
        df["trade_date"] = df["date"]

    # 归属到周五 reb_date
    df["reb_date"] = df["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

    # 每个 reb_date/ticker 取最后一个可用信号
    df = (
        df.sort_values(["reb_date", "ticker", "trade_date", "date"])
          .groupby(["reb_date", "ticker"], as_index=False)
          .tail(1)
    )

    # 截面排名
    df["rank"] = df.groupby("reb_date")["sig"].rank(method="average", pct=True)

    long_mask  = df["rank"] >= (1 - top_q)
    short_mask = df["rank"] <= top_q

    df["w"] = 0.0
    df.loc[long_mask,  "w"] =  1.0
    df.loc[short_mask, "w"] = -1.0

    # === 归一化：不使用 groupby.apply，避免 reb_date 丢失，也更快更稳 ===
    pos_cnt = df.groupby("reb_date")["w"].transform(lambda x: (x > 0).sum())
    neg_cnt = df.groupby("reb_date")["w"].transform(lambda x: (x < 0).sum())

    df.loc[df["w"] > 0, "w"] = df.loc[df["w"] > 0, "w"] / pos_cnt[df["w"] > 0]
    df.loc[df["w"] < 0, "w"] = df.loc[df["w"] < 0, "w"] / neg_cnt[df["w"] < 0]

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


def apply_cost_series(ret: pd.Series, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    to = 0.5 * weights.diff().abs().sum(axis=1)
    cost = to * (cost_bps / 10000.0)
    cost = cost.reindex(ret.index).fillna(0.0)
    return ret - cost


def sharpe(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(mu / sd * np.sqrt(ann_factor))


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) == 0:
        return np.nan
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def has_parquet_files(root: str) -> bool:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".parquet"):
                return True
    return False


def main():
    os.makedirs("results", exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    panel = panel.sort_values(["date", "ticker"])

    # 未来5日收益标签（用于日频 RankIC）
    fwd = panel[["date", "ticker", "fwd_ret_5d"]].copy()

    # seed signals
    sig_seed = pd.read_parquet(SIGNALS_PATH)
    sig_seed["date"] = pd.to_datetime(sig_seed["date"])
    sig_seed["ticker"] = sig_seed["ticker"].astype(str)
    sig_seed["alpha"] = sig_seed["alpha"].astype(str)
    sig_seed["signal"] = pd.to_numeric(sig_seed["signal"], errors="coerce")

    LLM_SIGNALS_DIR = "data/processed/signals_llm"

    # LLM signals (optional)
    if os.path.exists(LLM_SIGNALS_DIR) and has_parquet_files(LLM_SIGNALS_DIR):
        sig_llm = pd.read_parquet(LLM_SIGNALS_DIR)
        if "date" not in sig_llm.columns or "ticker" not in sig_llm.columns:
            sig_llm = sig_llm.reset_index()

        if "date" not in sig_llm.columns:
            if "level_0" in sig_llm.columns:
                sig_llm = sig_llm.rename(columns={"level_0": "date"})
            elif "index" in sig_llm.columns:
                sig_llm = sig_llm.rename(columns={"index": "date"})
        if "ticker" not in sig_llm.columns and "level_1" in sig_llm.columns:
            sig_llm = sig_llm.rename(columns={"level_1": "ticker"})

        required_cols = {"date", "ticker", "alpha", "signal"}
        missing = required_cols - set(sig_llm.columns)
        if missing:
            raise ValueError(f"LLM signals missing columns: {missing}. got columns={list(sig_llm.columns)}")

        sig_llm["date"] = pd.to_datetime(sig_llm["date"])
        sig_llm["ticker"] = sig_llm["ticker"].astype(str)
        sig_llm["alpha"] = sig_llm["alpha"].astype(str)
        sig_llm["signal"] = pd.to_numeric(sig_llm["signal"], errors="coerce")

        sig = pd.concat([sig_seed, sig_llm], ignore_index=True)
        sig = sig.drop_duplicates(subset=["date", "ticker", "alpha"], keep="last")

        print(f"Loaded signals: seed={sig_seed['alpha'].nunique()} llm={sig_llm['alpha'].nunique()} total={sig['alpha'].nunique()}")
    else:
        sig = sig_seed
        print(f"Loaded signals: seed={sig_seed['alpha'].nunique()} (no llm parquet found)")

    # 合并标签（日频 RankIC）
    sig = sig.merge(fwd, on=["date", "ticker"], how="left")

    # 周收益：日收益复利到周五
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0
    # 周五权重 -> 下一周收益（避免前视偏差）
    week_ret_fwd = week_ret.shift(-1)

    rows = []
    alpha_weekly_returns = {}  # alpha -> weekly long-short return series（用于相关性）

    for alpha, g in sig.groupby("alpha"):
        # --- 日频 RankIC(5d) ---
        def day_rankic(df_day):
            s = df_day["signal"]
            y = df_day["fwd_ret_5d"]
            valid = s.notna() & y.notna()
            if valid.sum() < 10:
                return np.nan
            return spearman_corr(s[valid], y[valid])

        rankic_ts = g.groupby("date", sort=True).apply(day_rankic)
        rankic_ts.index = pd.to_datetime(rankic_ts.index)

        mean_rankic = float(rankic_ts.mean(skipna=True))
        std_rankic  = float(rankic_ts.std(skipna=True, ddof=1))
        icir = mean_rankic / (std_rankic + 1e-12)
        stab = monthly_pos_ratio(rankic_ts)

        # --- 周频 RankIC(1w)：与回测一致（reb_date -> 下一周收益） ---
        sig_series_raw = g.set_index(["date", "ticker"])["signal"].sort_index()

        df_sig = sig_series_raw.rename("signal").reset_index()
        df_sig["date"] = pd.to_datetime(df_sig["date"])
        df_sig["ticker"] = df_sig["ticker"].astype(str)

        if TRADE_LAG_BDAYS and TRADE_LAG_BDAYS > 0:
            df_sig["trade_date"] = df_sig["date"] + BDay(TRADE_LAG_BDAYS)
        else:
            df_sig["trade_date"] = df_sig["date"]

        df_sig["reb_date"] = df_sig["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

        df_sig = (
            df_sig.sort_values(["reb_date", "ticker", "trade_date", "date"])
                 .groupby(["reb_date", "ticker"], as_index=False)
                 .tail(1)
        )

        wk_next = week_ret_fwd.stack().rename("wk_ret").reset_index()
        wk_next.columns = ["reb_date", "ticker", "wk_ret"]

        tmp = df_sig[["reb_date", "ticker", "signal"]].merge(
            wk_next, on=["reb_date", "ticker"], how="inner"
        )

        def one_week_ic(df_w):
            valid = df_w["signal"].notna() & df_w["wk_ret"].notna()
            if valid.sum() < 10:
                return np.nan
            return spearman_corr(df_w.loc[valid, "signal"], df_w.loc[valid, "wk_ret"])

        weekly_ic_ts = tmp.groupby("reb_date", sort=True).apply(one_week_ic)
        weekly_ic_ts.index = pd.to_datetime(weekly_ic_ts.index)

        mean_week_ic = float(weekly_ic_ts.mean(skipna=True))
        std_week_ic = float(weekly_ic_ts.std(skipna=True, ddof=1))
        icir_week = mean_week_ic / (std_week_ic + 1e-12)
        stab_week = monthly_pos_ratio(weekly_ic_ts)

        # ✅ direction：优先用周频 IC 的符号（与回测一致）
        if np.isfinite(mean_week_ic):
            direction = 1.0 if mean_week_ic >= 0 else -1.0
        else:
            direction = 1.0 if mean_rankic >= 0 else -1.0

        # --- 周频多空回测（用于 sharpe/mdd/turnover/corr） ---
        sig_series = direction * sig_series_raw
        weights = build_weekly_weights(sig_series, top_q=TOP_Q, trade_lag_bdays=TRADE_LAG_BDAYS)

        common_dates = weights.index.intersection(week_ret_fwd.index)
        weights = weights.loc[common_dates]
        wk = week_ret_fwd.loc[common_dates, weights.columns]

        ls_ret = (weights * wk).sum(axis=1).dropna()
        turnover_mean = compute_turnover(weights)
        ls_ret_net = apply_cost_series(ls_ret, weights, COST_BPS)

        sh = sharpe(ls_ret)
        sh_net = sharpe(ls_ret_net)

        nav = (1.0 + ls_ret.fillna(0.0)).cumprod()
        nav_net = (1.0 + ls_ret_net.fillna(0.0)).cumprod()
        mdd = max_drawdown(nav)
        mdd_net = max_drawdown(nav_net)

        rows.append({
            "alpha": str(alpha),

            # 日频口径
            "mean_rankIC_5d": mean_rankic,
            "ICIR_5d": icir,
            "stability_pos_ratio": stab,

            # 周频口径（与回测一致）
            "mean_rankIC_1w": mean_week_ic,
            "ICIR_1w": icir_week,
            "stability_pos_ratio_1w": stab_week,

            # ✅ 输出 direction，供 selector / backtest 口径一致
            "direction": float(direction),

            "turnover_weekly": turnover_mean,
            "sharpe_weekly": sh,
            "sharpe_weekly_net": sh_net,
            "mdd": mdd,
            "mdd_net": mdd_net,
            "n_rankic_days": int(rankic_ts.notna().sum()),
            "n_weeks": int(ls_ret.dropna().shape[0]),
        })

        alpha_weekly_returns[str(alpha)] = ls_ret.fillna(0.0)

    metrics = pd.DataFrame(rows).sort_values("mean_rankIC_1w", ascending=False)
    metrics.to_csv(OUT_METRICS, index=False)

    # --- Correlation matrix among alphas (weekly returns) ---
    ret_mat = pd.DataFrame(alpha_weekly_returns).sort_index()
    corr = ret_mat.corr()

    np.save(OUT_CORR, corr.to_numpy())

    # ✅ 保存 corr 的 alpha 顺序，selector 需要用它对齐
    with open(OUT_CORR_IDX, "w") as f:
        json.dump(list(ret_mat.columns), f, indent=2)

    print(f"Saved: {OUT_METRICS} | alphas={len(metrics)}")
    print(f"Saved: {OUT_CORR} | shape={corr.shape}")
    print(f"Saved: {OUT_CORR_IDX} | n={len(ret_mat.columns)}")

    print("\nTop 5 by mean_rankIC_1w (weekly, aligned with backtest):")
    print(
        metrics.head(5)[[
            "alpha",
            "mean_rankIC_1w",
            "ICIR_1w",
            "stability_pos_ratio_1w",
            "turnover_weekly",
            "sharpe_weekly_net",
            "mdd_net",
            "direction",
        ]]
    )


if __name__ == "__main__":
    main()