#!/usr/bin/env python3
# 09b_apply_llm_weights_backtest_regime.py
# LLM/Bayes alpha weights -> combine signals -> regime-conditional portfolio construction
# + optional EMA smoothing (dynamic by regime)
# + weak-signal filter
# + vol targeting position scaling (to reduce MDD / stabilize sharpe)

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


# -----------------------------
# basic utils / metrics
# -----------------------------
def try_print_torch_info():
    try:
        import torch as _torch
        print(f"[INFO] torch: {_torch.__version__}")
        print(f"[INFO] cuda available: {_torch.cuda.is_available()}")
    except Exception:
        print("[INFO] torch not available (ok).")


def sharpe(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float(mu / sd * np.sqrt(ann_factor))


def sortino(x: pd.Series, ann_factor=52, mar=0.0) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    downside = x[x < mar]
    if len(downside) < 5:
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


def var_cvar(x: pd.Series, q=0.05) -> Tuple[float, float]:
    x = x.dropna()
    if len(x) < 10:
        return float("nan"), float("nan")
    v = np.quantile(x, q)
    c = x[x <= v].mean() if (x <= v).any() else float("nan")
    return float(v), float(c)


def compute_turnover_pair(prev_w: pd.Series, curr_w: pd.Series) -> float:
    """
    turnover_t = 0.5 * sum |w_t - w_{t-1}|
    prev/curr must be aligned on the same index
    """
    if prev_w is None:
        return 0.0
    a = prev_w.fillna(0.0).to_numpy(dtype=float)
    b = curr_w.fillna(0.0).to_numpy(dtype=float)
    return float(0.5 * np.abs(a - b).sum())


# -----------------------------
# signal transforms
# -----------------------------
def winsorize_by_date(df: pd.DataFrame, col: str, lo=0.01, hi=0.99) -> pd.DataFrame:
    df = df.copy()

    def _clip(g):
        a = g[col].to_numpy()
        if len(a) < 10:
            return g
        lo_v = np.nanquantile(a, lo)
        hi_v = np.nanquantile(a, hi)
        g[col] = g[col].clip(lo_v, hi_v)
        return g

    return df.groupby("date", group_keys=False).apply(_clip)


def zscore_by_date(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()

    def _z(g):
        x = g[col].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd <= 1e-12:
            g[out_col] = 0.0
        else:
            g[out_col] = (x - mu) / sd
        return g

    return df.groupby("date", group_keys=False).apply(_z)


def clip_series(df: pd.DataFrame, col: str, z: float) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].clip(-z, z)
    return df


# -----------------------------
# JSON loaders
# -----------------------------
def load_infer_json(path: str) -> Tuple[List[str], Dict[str, float]]:
    with open(path, "r") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("infer_json must be a JSON object")

    selected = obj.get("selected", [])
    weights = obj.get("weights", {})

    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        raise ValueError("infer_json['selected'] must be a list[str]")
    if not isinstance(weights, dict):
        raise ValueError("infer_json['weights'] must be a dict")

    w = {}
    for a in selected:
        try:
            v = float(weights.get(a, 0.0))
        except Exception:
            v = 0.0
        if not np.isfinite(v) or v < 0:
            v = 0.0
        w[a] = float(v)

    s = sum(w.values())
    if s <= 0 and len(selected) > 0:
        w = {a: 1.0 / len(selected) for a in selected}
    elif s > 0:
        w = {a: float(v / s) for a, v in w.items()}

    return selected, w


def load_direction_from_metrics(metrics_csv: Optional[str], alphas: List[str]) -> Dict[str, float]:
    out = {a: 1.0 for a in alphas}
    if not metrics_csv or not os.path.exists(metrics_csv):
        return out

    try:
        m = pd.read_csv(metrics_csv)
    except Exception:
        return out

    if "alpha" not in m.columns:
        return out

    m["alpha"] = m["alpha"].astype(str)

    sign_col = None
    for c in ["sharpe_net", "sharpe", "ic_mean", "ic", "ret_mean"]:
        if c in m.columns:
            sign_col = c
            break
    if sign_col is None:
        return out

    mm = m.set_index("alpha")[sign_col].to_dict()
    for a in alphas:
        v = mm.get(a, None)
        try:
            v = float(v)
        except Exception:
            v = None
        if v is None or not np.isfinite(v) or abs(v) < 1e-12:
            out[a] = 1.0
        else:
            out[a] = 1.0 if v > 0 else -1.0
    return out


# -----------------------------
# Regime detection (simple, no external index needed)
# -----------------------------
def compute_weekly_market_series(panel: pd.DataFrame, reb_freq: str, trade_lag_bdays: int) -> pd.DataFrame:
    """
    Build a weekly "market" return series from panel by equal-weighting daily returns,
    then resampling to weekly return aligned to reb_freq.
    Returns df with index=reb_date (normalized and lagged same as weights), columns:
      mkt_week_ret, mkt_week_vol_ann, mkt_trend_ann
    """
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["ret_1d"] = pd.to_numeric(p["ret_1d"], errors="coerce")
    p = p.dropna(subset=["date", "ticker", "ret_1d"])

    # daily equal-weight "market"
    mkt_d = p.groupby("date")["ret_1d"].mean().sort_index()

    # weekly returns aligned to reb_freq
    mkt_w = (1.0 + mkt_d).resample(reb_freq).prod() - 1.0
    mkt_w = mkt_w.to_frame("mkt_week_ret")

    # align index to weights' reb_date convention
    mkt_w.index = pd.to_datetime(mkt_w.index).normalize()
    if trade_lag_bdays and trade_lag_bdays > 0:
        mkt_w.index = mkt_w.index + BDay(int(trade_lag_bdays))

    return mkt_w


def detect_regime_simple(
    mkt_week_ret: pd.Series,
    vol_lookback: int = 52,
    trend_lookback: int = 26,
    vol_thr: float = 0.22,
    trend_thr: float = 0.10,
) -> pd.Series:
    """
    Simple regime:
      - volatile: rolling ann vol > vol_thr
      - trending: |rolling ann mean| > trend_thr and not volatile
      - choppy: otherwise
    """
    r = mkt_week_ret.dropna().copy()
    if r.empty:
        return pd.Series(dtype=str)

    vol = r.rolling(vol_lookback, min_periods=max(10, vol_lookback // 3)).std(ddof=1) * np.sqrt(52)
    trend = r.rolling(trend_lookback, min_periods=max(10, trend_lookback // 3)).mean() * 52  # ann mean

    regime = pd.Series(index=r.index, dtype=str)
    regime.loc[:] = "choppy"
    regime.loc[vol > vol_thr] = "volatile"
    regime.loc[(vol <= vol_thr) & (trend.abs() > trend_thr)] = "trending"
    return regime


# -----------------------------
# Portfolio construction (dynamic by regime)
# -----------------------------
def build_reb_table_from_score(
    score_long: pd.DataFrame,
    reb_freq: str,
    trade_lag_bdays: int,
) -> pd.DataFrame:
    """
    Convert daily (date,ticker,score) to weekly reb table:
      columns: reb_date, ticker, score
    keep last score in that week for each ticker
    """
    df = score_long[["date", "ticker", "score"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["reb_date"] = df["date"].dt.to_period(reb_freq).dt.end_time
    df["reb_date"] = pd.to_datetime(df["reb_date"]).dt.normalize()
    if trade_lag_bdays and trade_lag_bdays > 0:
        df["reb_date"] = df["reb_date"] + BDay(int(trade_lag_bdays))

    df = df.sort_values(["reb_date", "date"])
    df = df.groupby(["reb_date", "ticker"], as_index=False).tail(1)
    return df[["reb_date", "ticker", "score"]]


def make_weights_one_reb(
    g: pd.DataFrame,
    top_q: float,
    long_short: bool,
    min_names: int,
    min_abs_z: float,
) -> pd.DataFrame:
    """
    g columns: ticker, score
    returns: ticker, w
    """
    g = g.dropna(subset=["score"]).copy()
    if g.empty or g["ticker"].nunique() < min_names:
        return pd.DataFrame(columns=["ticker", "w"])

    # zscore within reb_date for weak-signal filter
    s = g["score"].astype(float)
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.DataFrame(columns=["ticker", "w"])

    z = (s - mu) / sd
    g["z"] = z

    if float(min_abs_z) > 0:
        # If both tails are weak, skip trading this reb_date (return empty)
        if (g["z"].abs().max() < float(min_abs_z)):
            return pd.DataFrame(columns=["ticker", "w"])

    g = g.sort_values("score", ascending=False)
    n = len(g)
    tq = float(top_q)
    tq = 0.2 if (tq <= 0 or tq >= 0.5) else tq
    k = max(1, int(np.floor(n * tq)))

    if long_short:
        long_leg = g.head(k).copy()
        short_leg = g.tail(k).copy()
        long_leg["w"] = 1.0 / max(1, len(long_leg))
        short_leg["w"] = -1.0 / max(1, len(short_leg))
        out = pd.concat([long_leg[["ticker", "w"]], short_leg[["ticker", "w"]]], ignore_index=True)
    else:
        long_leg = g.head(k).copy()
        long_leg["w"] = 1.0 / max(1, len(long_leg))
        out = long_leg[["ticker", "w"]].copy()

    return out


def pivot_weights(weights_long: pd.DataFrame) -> pd.DataFrame:
    if weights_long is None or len(weights_long) == 0:
        return pd.DataFrame()
    w = (
        weights_long.pivot_table(index="reb_date", columns="ticker", values="w", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
        .astype(float)
    )
    return w


def smooth_weights_ema_dynamic(weights: pd.DataFrame, eta_by_date: pd.Series) -> pd.DataFrame:
    """
    eta_by_date indexed by weights.index (reb_date).
    w_t = eta_t * w_{t-1} + (1-eta_t) * w_target_t
    """
    if weights is None or weights.empty:
        return pd.DataFrame()

    W = weights.fillna(0.0).astype(float).sort_index().copy()
    eta = eta_by_date.reindex(W.index).fillna(0.0).astype(float)
    eta = eta.clip(lower=0.0, upper=0.999)

    out = W.copy()
    prev = None
    for dt in W.index:
        e = float(eta.loc[dt])
        if prev is None:
            out.loc[dt] = W.loc[dt]
        else:
            out.loc[dt] = e * prev + (1.0 - e) * W.loc[dt]
        prev = out.loc[dt].to_numpy(dtype=float, copy=True)
    return out


# -----------------------------
# Backtest simulator (supports regime leverage & vol targeting)
# -----------------------------
def simulate_weekly_portfolio(
    weights_target: pd.DataFrame,
    wk_ret: pd.DataFrame,
    cost_bps: float,
    slippage_bps: float,
    regime_by_date: pd.Series,
    gross_leverage_by_date: pd.Series,
    vol_target_by_date: pd.Series,
    vol_lookback: int,
    vol_scale_min: float,
    vol_scale_max: float,
) -> pd.DataFrame:
    """
    Iterate weeks:
      - apply gross leverage
      - apply vol targeting scale (based on past net returns of this same strategy)
      - compute turnover+costs on the *scaled* weights
    Output columns: date, gross_ret, net_ret, turnover, nav_gross, nav_net, regime, lev, vol_scale
    """
    if weights_target.empty:
        raise ValueError("weights_target empty")

    dates = weights_target.index.intersection(wk_ret.index)
    if len(dates) < 10:
        raise ValueError(f"Too few aligned weeks: {len(dates)}")

    W = weights_target.loc[dates].copy()
    R = wk_ret.loc[dates].reindex(columns=W.columns).fillna(0.0).copy()

    regime = regime_by_date.reindex(dates).fillna("choppy")
    lev = gross_leverage_by_date.reindex(dates).fillna(1.0).astype(float)
    vt = vol_target_by_date.reindex(dates).fillna(np.nan).astype(float)

    prev_w = None
    nav_g, nav_n = 1.0, 1.0
    hist_net = []

    rows = []
    for dt in dates:
        w_raw = W.loc[dt].fillna(0.0).astype(float)

        # regime leverage (gross exposure scaling)
        l = float(lev.loc[dt])
        w1 = w_raw * l

        # vol targeting scale (based on realized vol of net returns so far)
        vol_scale = 1.0
        target = vt.loc[dt]
        if np.isfinite(target) and len(hist_net) >= max(10, vol_lookback // 3):
            past = pd.Series(hist_net[-vol_lookback:])
            rv = float(past.std(ddof=1) * np.sqrt(52))
            if np.isfinite(rv) and rv > 1e-8:
                vol_scale = float(np.clip(target / rv, vol_scale_min, vol_scale_max))

        w = w1 * vol_scale

        # turnover + costs on scaled weights
        if prev_w is None:
            to = 0.0
        else:
            # align index
            idx = w.index.union(prev_w.index)
            to = compute_turnover_pair(prev_w.reindex(idx).fillna(0.0), w.reindex(idx).fillna(0.0))

        gross_ret = float((w * R.loc[dt]).sum())
        cost = to * (float(cost_bps) / 10000.0)
        slip = to * (float(slippage_bps) / 10000.0) if slippage_bps and slippage_bps > 0 else 0.0
        net_ret = gross_ret - cost - slip

        nav_g *= (1.0 + gross_ret)
        nav_n *= (1.0 + net_ret)

        hist_net.append(net_ret)
        prev_w = w.copy()

        rows.append({
            "date": dt,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": to,
            "nav_gross": nav_g,
            "nav_net": nav_n,
            "regime": str(regime.loc[dt]),
            "lev": l,
            "vol_scale": vol_scale,
            "vol_target": float(target) if np.isfinite(target) else np.nan,
        })

    return pd.DataFrame(rows)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--panel", type=str, required=True)
    ap.add_argument("--seed_signals", type=str, required=True)
    ap.add_argument("--llm_dir", type=str, default="data/processed/signals_llm")
    ap.add_argument("--infer_json", type=str, required=True)
    ap.add_argument("--metrics_csv", type=str, default=None)

    ap.add_argument("--reb_freq", type=str, default="W-FRI")
    ap.add_argument("--trade_lag_bdays", type=int, default=1)

    # base params (used if regime_mode=off)
    ap.add_argument("--top_q", type=float, default=0.2)
    ap.add_argument("--clip_z", type=float, default=6.0)
    ap.add_argument("--ema_eta", type=float, default=0.30)
    ap.add_argument("--min_abs_z", type=float, default=0.0)
    ap.add_argument("--min_names", type=int, default=10)
    ap.add_argument("--long_short", action="store_true", help="Enable long-short (default True)")
    ap.add_argument("--long_only", action="store_true", help="Force long-only")

    # pre-processing
    ap.add_argument("--winsorize", action="store_true")
    ap.add_argument("--winsor_lo", type=float, default=0.01)
    ap.add_argument("--winsor_hi", type=float, default=0.99)
    ap.add_argument("--zscore", action="store_true")

    # costs
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)

    # regime
    ap.add_argument("--regime_mode", type=str, default="simple", choices=["off", "simple"])
    ap.add_argument("--regime_vol_lookback", type=int, default=52)
    ap.add_argument("--regime_trend_lookback", type=int, default=26)
    ap.add_argument("--regime_vol_thr", type=float, default=0.22)
    ap.add_argument("--regime_trend_thr", type=float, default=0.10)

    # vol targeting
    ap.add_argument("--vol_target_mode", type=str, default="on", choices=["off", "on"])
    ap.add_argument("--vol_target_lookback", type=int, default=52)
    ap.add_argument("--vol_scale_min", type=float, default=0.4)
    ap.add_argument("--vol_scale_max", type=float, default=1.6)

    # output
    ap.add_argument("--out_nav", type=str, default="results/portfolio_nav_llmweights_regime.csv")
    ap.add_argument("--out_summary", type=str, default="results/portfolio_summary_llmweights_regime.txt")

    args = ap.parse_args()

    print("[INFO] Running backtest 09b (Regime + WeakSignalFilter + VolTarget + EMA)...")
    print(f"[INFO] python: {os.popen('python -V').read().strip()}")
    try_print_torch_info()

    for p in [args.panel, args.seed_signals, args.infer_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(os.path.dirname(args.out_nav) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    # long/short mode
    long_short = True
    if args.long_only:
        long_short = False
    elif args.long_short:
        long_short = True

    # ---------- load infer weights + direction ----------
    selected, w_alpha = load_infer_json(args.infer_json)
    direction = load_direction_from_metrics(args.metrics_csv, selected)
    print(f"[INFO] selected K={len(selected)}: {selected}")
    print(f"[INFO] weights: {w_alpha}")
    print(f"[INFO] direction(sign) from metrics: {direction}")

    # ---------- load signals ----------
    sig_seed = pd.read_parquet(args.seed_signals)
    sig_all = [sig_seed]

    if os.path.isdir(args.llm_dir):
        try:
            found = False
            for root, _, files in os.walk(args.llm_dir):
                if any(f.endswith(".parquet") for f in files):
                    found = True
                    break
            if found:
                sig_llm = pd.read_parquet(args.llm_dir)
                sig_all.append(sig_llm)
            else:
                print("[WARN] llm_dir exists but no parquet files found inside. Continue seed only.")
        except Exception as e:
            print(f"[WARN] Failed to read llm_dir: {e}. Continue seed only.")

    sig = pd.concat(sig_all, ignore_index=True)

    for c in ["alpha", "ticker"]:
        if c not in sig.columns:
            raise KeyError(f"signals missing required column: {c}. Have={list(sig.columns)}")
        sig[c] = sig[c].astype(str)

    if "date" not in sig.columns or "signal" not in sig.columns:
        raise KeyError(f"signals must contain ['date','signal']. Have={list(sig.columns)}")

    sig["date"] = pd.to_datetime(sig["date"])
    sig["signal"] = pd.to_numeric(sig["signal"], errors="coerce")
    sig = sig.dropna(subset=["date", "ticker", "alpha", "signal"])

    sig = sig[sig["alpha"].isin([str(a) for a in selected])].copy()
    if sig.empty:
        raise ValueError("No signals after filtering selected alphas. Check selected names vs signals alpha column.")

    # ---------- per-alpha transform + combine daily score ----------
    parts = []
    for a in selected:
        sa = sig[sig["alpha"] == a][["date", "ticker", "signal"]].copy()
        if sa.empty:
            continue

        if args.winsorize:
            sa = winsorize_by_date(sa.rename(columns={"signal": "x"}), "x", lo=args.winsor_lo, hi=args.winsor_hi)
            sa = sa.rename(columns={"x": "signal"})

        if args.zscore:
            sa = zscore_by_date(sa.rename(columns={"signal": "x"}), "x", "z")
            sa = sa.drop(columns=["x"]).rename(columns={"z": "signal"})

        if args.clip_z is not None:
            sa = clip_series(sa.rename(columns={"signal": "x"}), "x", float(args.clip_z))
            sa = sa.rename(columns={"x": "signal"})

        sa["signal"] = sa["signal"] * float(direction.get(a, 1.0)) * float(w_alpha.get(a, 0.0))
        parts.append(sa)

    if not parts:
        raise ValueError("All selected alphas had empty signals after processing.")

    comb = pd.concat(parts, ignore_index=True)
    score = comb.groupby(["date", "ticker"], as_index=False)["signal"].sum()
    score = score.rename(columns={"signal": "score"}).dropna()
    print(f"[INFO] score_long rows={len(score)} head:\n{score.head(3).to_string(index=False)}")

    # ---------- load panel (for returns + regime) ----------
    panel = pd.read_parquet(args.panel)
    if "date" not in panel.columns or "ticker" not in panel.columns or "ret_1d" not in panel.columns:
        raise KeyError(f"panel must contain ['date','ticker','ret_1d']. Have={list(panel.columns)}")

    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    panel["ret_1d"] = pd.to_numeric(panel["ret_1d"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker", "ret_1d"])

    # weekly stock returns matrix aligned to reb_freq
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(args.reb_freq).prod() - 1.0
    week_ret_fwd = week_ret.shift(-1)

    # align week_ret_fwd index to weights reb_date convention
    week_ret_fwd.index = pd.to_datetime(week_ret_fwd.index).normalize()
    if args.trade_lag_bdays and args.trade_lag_bdays > 0:
        week_ret_fwd.index = week_ret_fwd.index + BDay(int(args.trade_lag_bdays))

    # ---------- regime series ----------
    if args.regime_mode == "simple":
        mkt_w = compute_weekly_market_series(panel, args.reb_freq, args.trade_lag_bdays)
        regime = detect_regime_simple(
            mkt_week_ret=mkt_w["mkt_week_ret"],
            vol_lookback=args.regime_vol_lookback,
            trend_lookback=args.regime_trend_lookback,
            vol_thr=args.regime_vol_thr,
            trend_thr=args.regime_trend_thr,
        )
        print("[INFO] regime counts:", regime.value_counts(dropna=False).to_dict())
    else:
        regime = pd.Series(dtype=str)

    # ---------- default regime params (you can tune) ----------
    # 这些是“更偏实战”的默认值：熊/高波动更保守，震荡更严格过滤，趋势更进攻一点
    regime_params = {
        "trending": {"top_q": 0.22, "clip_z": 6.0, "ema_eta": 0.25, "min_abs_z": 0.15, "gross_lev": 1.00, "vol_target": 0.14},
        "choppy":   {"top_q": 0.16, "clip_z": 5.5, "ema_eta": 0.45, "min_abs_z": 0.35, "gross_lev": 0.90, "vol_target": 0.12},
        "volatile": {"top_q": 0.12, "clip_z": 4.5, "ema_eta": 0.60, "min_abs_z": 0.55, "gross_lev": 0.70, "vol_target": 0.10},
    }

    # if regime_mode=off, treat all as "trending" but override by args
    if args.regime_mode == "off":
        # all dates -> one bucket
        # (use args as fixed)
        pass

    # ---------- daily score -> reb table ----------
    reb_table = build_reb_table_from_score(score, args.reb_freq, args.trade_lag_bdays)
    if reb_table.empty:
        raise ValueError("reb_table empty (check score generation)")

    # ---------- build weights dynamically by regime ----------
    weights_long_rows = []
    eta_by_date = {}
    lev_by_date = {}
    vt_by_date = {}

    for dt, g in reb_table.groupby("reb_date"):
        if args.regime_mode == "simple":
            r = str(regime.get(dt, "choppy"))
        else:
            r = "trending"  # dummy bucket

        if args.regime_mode == "off":
            # fixed params from args
            top_q = float(args.top_q)
            clip_z = float(args.clip_z) if args.clip_z is not None else 0.0
            ema_eta = float(args.ema_eta) if args.ema_eta is not None else 0.0
            min_abs_z = float(args.min_abs_z)
            gross_lev = 1.0
            vol_target = np.nan
        else:
            p = regime_params.get(r, regime_params["choppy"])
            top_q = float(p["top_q"])
            clip_z = float(p["clip_z"])
            ema_eta = float(p["ema_eta"])
            min_abs_z = float(p["min_abs_z"])
            gross_lev = float(p["gross_lev"])
            vol_target = float(p["vol_target"])

        # apply clip_z at this reb_date on cross-section (simple: just clip scores)
        gg = g.copy()
        if clip_z and clip_z > 0:
            # clip by zscore
            s = gg["score"].astype(float)
            mu = float(s.mean())
            sd = float(s.std(ddof=0))
            if np.isfinite(sd) and sd > 1e-12:
                z = (s - mu) / sd
                z = z.clip(-clip_z, clip_z)
                gg["score"] = (z * sd) + mu

        w_one = make_weights_one_reb(
            gg[["ticker", "score"]],
            top_q=top_q,
            long_short=long_short,
            min_names=int(args.min_names),
            min_abs_z=min_abs_z,
        )

        if not w_one.empty:
            w_one = w_one.copy()
            w_one.insert(0, "reb_date", pd.to_datetime(dt))
            weights_long_rows.append(w_one)

        eta_by_date[pd.to_datetime(dt)] = ema_eta
        lev_by_date[pd.to_datetime(dt)] = gross_lev
        vt_by_date[pd.to_datetime(dt)] = vol_target

    weights_long = pd.concat(weights_long_rows, ignore_index=True) if weights_long_rows else pd.DataFrame()
    weights = pivot_weights(weights_long)
    if weights.empty:
        raise ValueError("weights empty after dynamic construction (too strict min_abs_z or min_names?)")

    # ---------- EMA smoothing (dynamic by regime) ----------
    eta_series = pd.Series(eta_by_date).sort_index()
    weights_sm = smooth_weights_ema_dynamic(weights, eta_series)

    # ---------- align weekly returns to weights columns ----------
    common_dates = weights_sm.index.intersection(week_ret_fwd.index)
    weights_sm = weights_sm.loc[common_dates].sort_index()
    wk = week_ret_fwd.loc[common_dates].reindex(columns=weights_sm.columns).fillna(0.0)

    # ---------- backtest simulate with leverage + vol targeting ----------
    lev_series = pd.Series(lev_by_date).sort_index()
    vt_series = pd.Series(vt_by_date).sort_index()

    if args.vol_target_mode == "off":
        vt_series = vt_series * np.nan

    nav = simulate_weekly_portfolio(
        weights_target=weights_sm,
        wk_ret=wk,
        cost_bps=float(args.cost_bps),
        slippage_bps=float(args.slippage_bps),
        regime_by_date=regime,
        gross_leverage_by_date=lev_series,
        vol_target_by_date=vt_series,
        vol_lookback=int(args.vol_target_lookback),
        vol_scale_min=float(args.vol_scale_min),
        vol_scale_max=float(args.vol_scale_max),
    )

    # ---------- summary ----------
    gross = pd.Series(nav["gross_ret"].values, index=pd.to_datetime(nav["date"]))
    net = pd.Series(nav["net_ret"].values, index=pd.to_datetime(nav["date"]))
    to = pd.Series(nav["turnover"].values, index=pd.to_datetime(nav["date"]))
    nav_net = pd.Series(nav["nav_net"].values, index=pd.to_datetime(nav["date"]))
    nav_gross = pd.Series(nav["nav_gross"].values, index=pd.to_datetime(nav["date"]))

    ann_return_net = float(net.mean() * 52)
    ann_vol_net = float(net.std(ddof=1) * np.sqrt(52))
    v5, c5 = var_cvar(net, q=0.05)

    net_s = net.dropna()
    win_rate = float((net_s > 0).mean()) if len(net_s) else float("nan")
    avg_win = float(net_s[net_s > 0].mean()) if (net_s > 0).any() else float("nan")
    avg_loss = float(net_s[net_s < 0].mean()) if (net_s < 0).any() else float("nan")
    avg_win_loss_ratio = (
        float(avg_win / abs(avg_loss))
        if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0
        else float("nan")
    )
    profit_factor = (
        float(net_s[net_s > 0].sum() / abs(net_s[net_s < 0].sum()))
        if (net_s > 0).any() and (net_s < 0).any() and net_s[net_s < 0].sum() != 0
        else float("nan")
    )
    sign = np.sign(net_s)
    max_win_streak = int((sign.gt(0)).astype(int).groupby((sign.le(0)).cumsum()).sum().max()) if len(net_s) else 0
    max_loss_streak = int((sign.lt(0)).astype(int).groupby((sign.ge(0)).cumsum()).sum().max()) if len(net_s) else 0

    regime_counts = nav["regime"].value_counts(dropna=False).to_dict()

    summary = {
        "mode": "09b_regime_voltarget_ema",
        "alphas_used": selected,
        "alpha_weights": w_alpha,
        "direction": direction,
        "rebalance": args.reb_freq,
        "TRADE_LAG_BDAYS": int(args.trade_lag_bdays),
        "weeks": int(len(net)),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "sortino_net": sortino(net),
        "ann_return_net": ann_return_net,
        "ann_vol_net": ann_vol_net,
        "mdd_gross": max_drawdown(nav_gross),
        "mdd_net": max_drawdown(nav_net),
        "avg_turnover": float(to.mean()),
        "cost_bps": float(args.cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "winsorize": bool(args.winsorize),
        "zscore": bool(args.zscore),
        "vol_target_mode": args.vol_target_mode,
        "vol_target_lookback": int(args.vol_target_lookback),
        "vol_scale_min": float(args.vol_scale_min),
        "vol_scale_max": float(args.vol_scale_max),
        "regime_mode": args.regime_mode,
        "regime_counts": regime_counts,
        "var_5pct_net": v5,
        "cvar_5pct_net": c5,
        "skew_net": float(net_s.skew()) if len(net_s) > 10 else float("nan"),
        "kurt_net": float(net_s.kurtosis()) if len(net_s) > 10 else float("nan"),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }

    # ---------- save ----------
    nav_out = nav.copy()
    nav_out["date"] = pd.to_datetime(nav_out["date"]).dt.strftime("%Y-%m-%d")
    nav_out.to_csv(args.out_nav, index=False)

    with open(args.out_summary, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] Saved: {args.out_nav}")
    print(f"[OK] Saved: {args.out_summary}")
    print("[INFO] Summary (key):")
    for k in [
        "weeks", "sharpe_net", "ann_return_net", "ann_vol_net", "mdd_net", "avg_turnover",
        "win_rate", "profit_factor", "avg_win_loss_ratio", "max_win_streak", "max_loss_streak",
        "regime_counts"
    ]:
        print(f"  {k}: {summary.get(k)}")


if __name__ == "__main__":
    main()