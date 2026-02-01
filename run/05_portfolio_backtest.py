#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_portfolio_backtest.py

Multi-alpha portfolio backtest (weekly rebalance, aligned with selector + metrics logic):
- Reads selected alphas (v3 -> v2 -> legacy)
- Loads seed signals + optional LLM signals (partitioned parquet dir)
- Builds daily combined score per (date,ticker) from selected alphas
- Optional: winsorize per date (cross-section)
- Builds weekly long/short equal-weight portfolio (TOP_Q / -TOP_Q)
- Uses "Friday signal -> next week return" to avoid lookahead
- Costs: turnover-based (optionally separate long/short costs)
- Saves: results/portfolio_nav.csv, results/portfolio_summary.txt
- Optional checkpoints: results/checkpoints/score.parquet, weights.parquet

Run from project root (basicquant) so relative paths work:
  cd /home/users/ntu/shuyan00/scratch/nscc_qwen/basicquant
  python /home/users/ntu/shuyan00/scratch/nscc_qwen/run/05_portfolio_backtest.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


# -----------------------------
# Default Paths (relative to project root)
# -----------------------------
PANEL_PATH      = "data/processed/panel.parquet"
SIGNALS_PATH    = "data/processed/signals.parquet"          # seed signals
LLM_SIGNALS_DIR = "data/processed/signals_llm"              # optional partitioned parquet dataset

SELECT_CANDIDATES = [
    "/home/users/ntu/shuyan00/scratch/nscc_qwen/basicquant/results/selected_alphas_v3.json",
    "/home/users/ntu/shuyan00/scratch/nscc_qwen/basicquant/results/selected_alphas_v2.json",
    "/home/users/ntu/shuyan00/scratch/nscc_qwen/basicquant/results/selected_alphas.json",
]
OUT_NAV     = "results/portfolio_nav.csv"
OUT_SUMMARY = "results/portfolio_summary.txt"
CKPT_DIR    = "results/checkpoints"


# -----------------------------
# Utilities
# -----------------------------
def has_parquet_files(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".parquet"):
                return True
    return False


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def sharpe(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float(mu / sd * np.sqrt(ann_factor))


def sortino(x: pd.Series, mar=0.0, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 30:
        return float("nan")
    downside = x[x < mar]
    if len(downside) < 10:
        return float("nan")
    dd = downside.std(ddof=1)
    if dd <= 1e-12:
        return float("nan")
    return float((x.mean() - mar) / dd * np.sqrt(ann_factor))


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) == 0:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def annual_return(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.mean() * ann_factor)


def annual_vol(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.std(ddof=1) * np.sqrt(ann_factor))


def calmar(x: pd.Series, nav: pd.Series) -> float:
    ar = annual_return(x)
    mdd = abs(max_drawdown(nav))
    if (not np.isfinite(ar)) or (not np.isfinite(mdd)) or mdd <= 1e-12:
        return float("nan")
    return float(ar / mdd)


def var_cvar(x: pd.Series, q=0.05):
    x = x.dropna()
    if len(x) < 50:
        return float("nan"), float("nan")
    v = float(x.quantile(q))
    tail = x[x <= v]
    cv = float(tail.mean()) if len(tail) else float("nan")
    return v, cv


def print_alignment_report(weights: pd.DataFrame, week_ret_fwd: pd.DataFrame):
    w_dates = weights.index
    r_dates = week_ret_fwd.index
    common = w_dates.intersection(r_dates)
    miss_w = len(w_dates.difference(r_dates))
    miss_r = len(r_dates.difference(w_dates))
    print(f"[ALIGN] weights dates={len(w_dates)} week_ret_fwd dates={len(r_dates)} common={len(common)}")
    if miss_w > 0:
        ex = list(w_dates.difference(r_dates)[:3])
        print(f"[WARN] {miss_w} weight dates missing returns -> dropped. e.g. {ex}")
    if miss_r > 0:
        print(f"[INFO] {miss_r} return dates missing weights -> ignored.")


def load_selected_alphas() -> list[str]:
    select_path = None
    for p in SELECT_CANDIDATES:
        if os.path.exists(p):
            select_path = p
            break
    if select_path is None:
        raise FileNotFoundError(f"No selection json found. Tried: {SELECT_CANDIDATES}. Run selector first.")
    print(f"[Selection] using {os.path.abspath(select_path)}")

    with open(select_path, "r") as f:
        sel = json.load(f)

    if isinstance(sel, dict) and "selected" in sel:
        # v2/v3 formats
        selected = []
        for x in sel["selected"]:
            if isinstance(x, dict) and "alpha" in x:
                selected.append(str(x["alpha"]))
            else:
                selected.append(str(x))
    elif isinstance(sel, list):
        selected = [str(x) for x in sel]
    else:
        raise ValueError(f"Unknown selection json format in {select_path}")

    # de-dup preserve order
    out = []
    seen = set()
    for a in selected:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def winsorize_by_date(score_df: pd.DataFrame, lower=0.01, upper=0.99) -> pd.DataFrame:
    """
    Cross-sectional winsorization each date.
    score_df: columns [date,ticker,score]
    """
    df = score_df.copy()
    def _clip(g):
        if g["score"].notna().sum() < 5:
            return g
        lo = g["score"].quantile(lower)
        hi = g["score"].quantile(upper)
        g["score"] = g["score"].clip(lo, hi)
        return g
    return df.groupby("date", group_keys=False).apply(_clip)


# -----------------------------
# Portfolio construction
# -----------------------------
def build_weekly_weights_from_score(
    score_long: pd.DataFrame,
    top_q=0.2,
    trade_lag_bdays: int = 0,
    reb_freq: str = "W-FRI",
) -> pd.DataFrame:
    """
    Input:
      score_long: columns [date,ticker,score], higher=more long
    Output:
      weights: index=reb_date, columns=ticker, long sum=+1, short sum=-1, net=0
    """
    df = score_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["date", "ticker"])

    if trade_lag_bdays and trade_lag_bdays > 0:
        df["trade_date"] = df["date"] + BDay(trade_lag_bdays)
    else:
        df["trade_date"] = df["date"]

    # rebalance date bucket (Friday end_time normalized)
    # NOTE: works for W-FRI. If you change reb_freq, keep consistent with resample below.
    df["reb_date"] = df["trade_date"].dt.to_period(reb_freq).dt.end_time.dt.normalize()

    # keep last signal per reb_date/ticker
    df = (
        df.sort_values(["reb_date", "ticker", "trade_date", "date"])
          .groupby(["reb_date", "ticker"], as_index=False)
          .tail(1)
    )

    # cross-sectional rank per reb_date
    df["rank"] = df.groupby("reb_date")["score"].rank(method="average", pct=True)

    long_mask  = df["rank"] >= (1 - top_q)
    short_mask = df["rank"] <= top_q

    df["w_raw"] = 0.0
    df.loc[long_mask,  "w_raw"] =  1.0
    df.loc[short_mask, "w_raw"] = -1.0

    # vectorized normalization by reb_date
    pos_cnt = df["w_raw"].gt(0).groupby(df["reb_date"]).transform("sum").replace(0, np.nan)
    neg_cnt = df["w_raw"].lt(0).groupby(df["reb_date"]).transform("sum").replace(0, np.nan)

    df["w"] = 0.0
    df.loc[df["w_raw"] > 0, "w"] = (df.loc[df["w_raw"] > 0, "w_raw"] / pos_cnt[df["w_raw"] > 0]).astype(float)
    df.loc[df["w_raw"] < 0, "w"] = (df.loc[df["w_raw"] < 0, "w_raw"] / neg_cnt[df["w_raw"] < 0]).astype(float)

    df["w"] = df["w"].fillna(0.0)

    weights = df.pivot(index="reb_date", columns="ticker", values="w").fillna(0.0)
    weights.index.name = "date"
    return weights


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    # 0.5 * sum(|w_t - w_{t-1}|)
    dw = weights.diff().abs()
    return 0.5 * dw.sum(axis=1)


def compute_turnover_long_short(weights: pd.DataFrame):
    """
    Rough long/short leg turnover split.
    """
    w = weights.fillna(0.0)
    dw = w.diff().fillna(0.0)

    prev = w.shift(1).fillna(0.0)
    long_to  = 0.5 * dw.where(prev > 0, 0.0).abs().sum(axis=1)
    short_to = 0.5 * dw.where(prev < 0, 0.0).abs().sum(axis=1)
    return long_to, short_to


def simulate_slippage_placeholder(weights: pd.DataFrame, slippage_bps: float = 0.0) -> pd.Series:
    if slippage_bps is None or slippage_bps <= 0:
        return pd.Series(0.0, index=weights.index)
    to = compute_turnover(weights)
    return to * (slippage_bps / 10000.0)


# -----------------------------
# Signals loading + score
# -----------------------------
def load_signals(selected: list[str]) -> pd.DataFrame:
    sig_seed = pd.read_parquet(SIGNALS_PATH)
    sig_all = [sig_seed]

    if os.path.exists(LLM_SIGNALS_DIR) and has_parquet_files(LLM_SIGNALS_DIR):
        sig_llm = pd.read_parquet(LLM_SIGNALS_DIR)
        # normalize if index got stored
        if ("date" not in sig_llm.columns) or ("ticker" not in sig_llm.columns):
            sig_llm = sig_llm.reset_index()
        # normalize possible reset_index names
        if "date" not in sig_llm.columns and "level_0" in sig_llm.columns:
            sig_llm = sig_llm.rename(columns={"level_0": "date"})
        if "ticker" not in sig_llm.columns and "level_1" in sig_llm.columns:
            sig_llm = sig_llm.rename(columns={"level_1": "ticker"})
        sig_all.append(sig_llm)
        print(f"[INFO] Loaded LLM signals from {LLM_SIGNALS_DIR}")
    else:
        if os.path.exists(LLM_SIGNALS_DIR):
            print(f"[WARN] LLM signals dir exists but no parquet files found inside: {LLM_SIGNALS_DIR}")
        else:
            print(f"[INFO] No LLM signals dir found: {LLM_SIGNALS_DIR}")

    sig = pd.concat(sig_all, ignore_index=True)

    # schema normalize
    for col in ["alpha", "ticker"]:
        if col in sig.columns:
            sig[col] = sig[col].astype(str)
    sig["date"] = pd.to_datetime(sig["date"])
    sig["signal"] = pd.to_numeric(sig["signal"], errors="coerce")

    need = {"date", "ticker", "alpha", "signal"}
    missing = need - set(sig.columns)
    if missing:
        raise ValueError(f"Signals missing columns: {missing}. got={list(sig.columns)}")

    # keep only selected
    sel_set = set(map(str, selected))
    sig = sig[sig["alpha"].isin(sel_set)].copy()

    # drop duplicates
    sig = sig.drop_duplicates(subset=["date", "ticker", "alpha"], keep="last")
    return sig


def compute_score_equal_weight(sig: pd.DataFrame) -> pd.DataFrame:
    """
    Daily combined score per (date,ticker) = mean(signal across selected alphas).
    Returns long DF: [date,ticker,score]
    """
    score = sig.groupby(["date", "ticker"], as_index=False)["signal"].mean()
    score = score.rename(columns={"signal": "score"})
    return score


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_q", type=float, default=0.2)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--reb_freq", type=str, default="W-FRI")
    ap.add_argument("--trade_lag_bdays", type=int, default=0)

    ap.add_argument("--winsorize", action="store_true", help="enable cross-sectional winsorization per date")
    ap.add_argument("--winsor_lo", type=float, default=0.01)
    ap.add_argument("--winsor_hi", type=float, default=0.99)

    ap.add_argument("--save_ckpt", action="store_true", help="save score/weights checkpoints")
    ap.add_argument("--slippage_bps", type=float, default=0.0)

    ap.add_argument("--use_short_cost", action="store_true", help="use separate long/short costs")
    ap.add_argument("--long_cost_bps", type=float, default=10.0)
    ap.add_argument("--short_cost_bps", type=float, default=15.0)

    return ap.parse_args()


def main():
    args = parse_args()

    safe_mkdir("results")

    # 1) selected alphas
    selected = load_selected_alphas()
    print(f"[Selection] {len(selected)} alphas: {selected}")

    # 2) load signals
    sig = load_signals(selected)
    if sig.empty:
        raise RuntimeError("No signals after filtering selected alphas. Check selection json vs signals alpha names.")

    # 3) daily combined score
    score = compute_score_equal_weight(sig)

    if args.winsorize:
        score = winsorize_by_date(score, lower=args.winsor_lo, upper=args.winsor_hi)
        print(f"[INFO] winsorize enabled: lo={args.winsor_lo}, hi={args.winsor_hi}")

    # 4) weekly weights
    weights = build_weekly_weights_from_score(
        score_long=score,
        top_q=args.top_q,
        trade_lag_bdays=args.trade_lag_bdays,
        reb_freq=args.reb_freq
    )

    if args.save_ckpt:
        safe_mkdir(CKPT_DIR)
        score.to_parquet(os.path.join(CKPT_DIR, "score.parquet"), index=False)
        weights.to_parquet(os.path.join(CKPT_DIR, "weights.parquet"))
        print(f"[CKPT] saved to {CKPT_DIR}/ (score.parquet, weights.parquet)")

    # 5) load panel and compute weekly forward returns
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    panel = panel.sort_values(["date", "ticker"])

    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(args.reb_freq).prod() - 1.0
    week_ret_fwd = week_ret.shift(-1)  # Friday weights -> next week returns

    print_alignment_report(weights, week_ret_fwd)

    # align dates & tickers
    common_dates = weights.index.intersection(week_ret_fwd.index)
    weights = weights.loc[common_dates]

    # ensure columns exist in returns
    cols = [c for c in weights.columns if c in week_ret_fwd.columns]
    if len(cols) < 5:
        raise RuntimeError(f"Too few tickers matched between weights and returns. matched={len(cols)}")
    weights = weights[cols]
    wk = week_ret_fwd.loc[common_dates, cols]

    # 6) gross returns
    gross = (weights * wk).sum(axis=1).dropna()

    # 7) costs
    if args.use_short_cost:
        long_to, short_to = compute_turnover_long_short(weights)
        cost = long_to * (args.long_cost_bps / 10000.0) + short_to * (args.short_cost_bps / 10000.0)
        to_total = compute_turnover(weights)
        avg_to = float(to_total.mean())
    else:
        to_total = compute_turnover(weights)
        cost = to_total * (args.cost_bps / 10000.0)
        avg_to = float(to_total.mean())

    # slippage placeholder
    slip = simulate_slippage_placeholder(weights, slippage_bps=args.slippage_bps)
    cost = cost.reindex(gross.index).fillna(0.0) + slip.reindex(gross.index).fillna(0.0)

    net = gross - cost

    # sanity
    if (net <= -1.0).any() or (gross <= -1.0).any():
        bad_g = gross[gross <= -1.0]
        bad_n = net[net <= -1.0]
        print(f"[WARN] weekly return <= -100%. gross={len(bad_g)} weeks, net={len(bad_n)} weeks. Check data quality/alignment.")

    # nav
    nav_gross = (1.0 + gross.fillna(0.0)).cumprod()
    nav_net   = (1.0 + net.fillna(0.0)).cumprod()

    # risk stats
    var5, cvar5 = var_cvar(net, q=0.05)

    summary = {
        "alphas_used": selected,
        "TOP_Q": float(args.top_q),
        "rebalance": args.reb_freq,
        "TRADE_LAG_BDAYS": int(args.trade_lag_bdays),

        "weeks": int(len(net)),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "sortino_net": sortino(net),
        "ann_return_net": annual_return(net),
        "ann_vol_net": annual_vol(net),
        "calmar_net": calmar(net, nav_net),

        "mdd_gross": max_drawdown(nav_gross),
        "mdd_net": max_drawdown(nav_net),

        "avg_turnover": avg_to,
        "slippage_bps": float(args.slippage_bps),

        "use_short_cost": bool(args.use_short_cost),
        "cost_bps": float(args.cost_bps),
        "long_cost_bps": float(args.long_cost_bps),
        "short_cost_bps": float(args.short_cost_bps),

        "var_5pct_net": var5,
        "cvar_5pct_net": cvar5,
        "skew_net": float(net.dropna().skew()) if net.dropna().shape[0] > 30 else float("nan"),
        "kurt_net": float(net.dropna().kurtosis()) if net.dropna().shape[0] > 30 else float("nan"),
    }

    out = pd.DataFrame({
        "date": gross.index,
        "gross_ret": gross.values,
        "net_ret": net.values,
        "turnover": to_total.reindex(gross.index).fillna(0.0).values,
        "cost_total": cost.values,
        "nav_gross": nav_gross.values,
        "nav_net": nav_net.values,
    })
    out.to_csv(OUT_NAV, index=False)

    with open(OUT_SUMMARY, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved: {OUT_NAV}")
    print(f"Saved: {OUT_SUMMARY}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()