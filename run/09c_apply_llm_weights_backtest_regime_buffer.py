#!/usr/bin/env python3
# 09c_apply_llm_weights_backtest_regime_buffer.py
# Soft Regime + Buffer (hysteresis) + Score/Rank weighting + Turnover Controller (+ optional Vol Target)

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


# ---------------- utils ----------------
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


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    dw = weights.diff().abs()
    return 0.5 * dw.sum(axis=1)


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


def softmax_row(a: np.ndarray, temp: float = 1.0) -> np.ndarray:
    t = max(float(temp), 1e-6)
    x = a / t
    x = x - np.nanmax(x)
    e = np.exp(np.clip(x, -50, 50))
    s = np.nansum(e)
    if not np.isfinite(s) or s <= 1e-12:
        return np.ones_like(a) / len(a)
    return e / s


def renormalize_long_short(w: pd.Series) -> pd.Series:
    """Ensure long sum=+1, short sum=-1 (if exist)."""
    w = w.copy()
    pos = w[w > 0]
    neg = w[w < 0]
    if len(pos) > 0:
        s = pos.sum()
        if abs(s) > 1e-12:
            w.loc[pos.index] = pos / s
    if len(neg) > 0:
        s = neg.abs().sum()
        if abs(s) > 1e-12:
            w.loc[neg.index] = neg / s * (-1.0)
    return w


# ---------------- alpha io ----------------
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
    if s <= 0:
        w = {a: 1.0 / len(selected) for a in selected}
    else:
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


# ---------------- regime & alpha softmix ----------------
def alpha_type(name: str) -> str:
    s = str(name).lower()
    if "mom" in s:
        return "mom"
    if "rev" in s or "revert" in s:
        return "rev"
    if "vol" in s or "risk" in s:
        return "vol"
    return "other"


def make_regime_probs(
    panel: pd.DataFrame,
    reb_freq: str,
    trade_lag_bdays: int,
    # windows
    trend_window_w: int = 26,     # 趋势强度窗口（半年）
    ac_window_w: int = 26,        # 自相关/反转窗口（半年）
    vol_short_w: int = 13,        # 短期波动（季度）
    vol_long_w: int = 52,         # 长期波动（年）
    # soft controls
    temp: float = 1.0,
    prob_smooth_w: int = 3,       # 概率平滑（3周）
    zscore_window_w: int = 104,   # 分数标准化窗口（2年，足够稳）
) -> pd.DataFrame:
    """
    输出 index=reb_date, columns=[p_trending,p_choppy,p_volatile]
    只用 weekly market return 序列 wk 构造正交 scores，然后 z-score 对齐尺度，再 softmax。
    """

    # 1) weekly market return
    mkt = panel.groupby("date", as_index=True)["ret_1d"].mean().sort_index()
    wk = (1.0 + mkt).resample(reb_freq).prod() - 1.0
    wk.index = pd.to_datetime(wk.index).normalize()
    if trade_lag_bdays and trade_lag_bdays > 0:
        wk.index = wk.index + BDay(int(trade_lag_bdays))

    wk = wk.dropna()
    if wk.empty:
        return pd.DataFrame(columns=["p_trending", "p_choppy", "p_volatile"])

    eps = 1e-12

    # --------------------
    # 2) three orthogonal-ish scores (logits)
    # --------------------

    # (a) TREND strength: |mean| / std
    mu = wk.rolling(trend_window_w, min_periods=max(8, trend_window_w // 2)).mean()
    sd = wk.rolling(trend_window_w, min_periods=max(8, trend_window_w // 2)).std(ddof=1)
    s_trend = (mu.abs() / (sd + eps)).fillna(0.0)

    # (b) VOL shock: log(short_vol / long_vol)
    vol_s = wk.rolling(vol_short_w, min_periods=max(5, vol_short_w // 2)).std(ddof=1)
    vol_l = wk.rolling(vol_long_w, min_periods=max(10, vol_long_w // 2)).std(ddof=1)
    vol_ratio = (vol_s / (vol_l + eps)).clip(lower=eps)
    s_vol = np.log(vol_ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # (c) CHOP / mean-reversion: negative autocorr + sign flip
    def _ac1(x):
        x = pd.Series(x)
        if x.count() < 8:
            return 0.0
        v = x.autocorr(lag=1)
        return 0.0 if not np.isfinite(v) else float(v)

    ac1 = wk.rolling(ac_window_w, min_periods=max(10, ac_window_w // 2)).apply(_ac1, raw=False)
    mean_rev = (-ac1).clip(lower=0.0).fillna(0.0)  # 只取负自相关（反转）

    sign_flip = ((wk * wk.shift(1)) < 0).rolling(ac_window_w, min_periods=max(10, ac_window_w // 2)).mean()
    sign_flip = sign_flip.fillna(0.0)

    s_chop = (0.7 * mean_rev + 0.3 * sign_flip).fillna(0.0)

    # --------------------
    # 3) scale alignment (关键)：rolling z-score 让三个 logit 可比
    # --------------------
    def _rz(x: pd.Series) -> pd.Series:
        m = x.rolling(zscore_window_w, min_periods=max(20, zscore_window_w // 4)).mean()
        s = x.rolling(zscore_window_w, min_periods=max(20, zscore_window_w // 4)).std(ddof=1)
        z = (x - m) / (s + eps)
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)

    z_trend = _rz(s_trend)
    z_chop  = _rz(s_chop)
    z_vol   = _rz(s_vol)

    # --------------------
    # 4) softmax -> probabilities
    # --------------------
    probs = []
    idx = z_trend.index.intersection(z_chop.index).intersection(z_vol.index)
    for t in idx:
        a = np.array([z_trend.loc[t], z_chop.loc[t], z_vol.loc[t]], dtype=float)
        p = softmax_row(a, temp=temp)
        probs.append(p)

    probs = np.asarray(probs)
    out = pd.DataFrame(
        probs, index=idx, columns=["p_trending", "p_choppy", "p_volatile"]
    ).sort_index()

    # --------------------
    # 5) probability smoothing (避免抖动) + renorm
    # --------------------
    if prob_smooth_w and prob_smooth_w > 1:
        out = out.rolling(prob_smooth_w, min_periods=1).mean()
        s = out.sum(axis=1).replace(0, np.nan)
        out = out.div(s, axis=0).fillna(1.0 / 3.0)

    # --------------------
    # 6) sanity checks (护栏)：一眼发现门控失真
    # --------------------
    dom = out.idxmax(axis=1).value_counts(normalize=True)
    if dom.max() > 0.85:
        print(f"[WARN] Regime dominance too high: {dom.to_dict()}  -> check logits scale/design/temp")
    ent = -(out * np.log(out + eps)).sum(axis=1).mean()
    if ent < 0.6:  # 3类最大熵约 1.10
        print(f"[WARN] Regime entropy too low (too peaky): mean_entropy={ent:.3f}. Consider larger temp or better features.")

    return out


def make_alpha_weights_by_regime_soft(
    selected: List[str],
    base_w: Dict[str, float],
    regime_probs: pd.DataFrame,
    mult_trending: Dict[str, float],
    mult_choppy: Dict[str, float],
    mult_volatile: Dict[str, float],
) -> pd.DataFrame:
    """
    Return df index=reb_date columns=alpha, weights sum to 1 each date.
    base_w is prior weights (LLM/Bayes).
    multipliers based on alpha type.
    """
    alphas = [str(a) for a in selected]
    base = np.array([float(base_w.get(a, 0.0)) for a in alphas], dtype=float)
    if base.sum() <= 0:
        base = np.ones(len(alphas)) / len(alphas)
    else:
        base = base / base.sum()

    rows = []
    for dt, r in regime_probs.iterrows():
        p_t = float(r["p_trending"])
        p_c = float(r["p_choppy"])
        p_v = float(r["p_volatile"])

        eff = []
        for a in alphas:
            t = alpha_type(a)
            mt = float(mult_trending.get(t, 1.0))
            mc = float(mult_choppy.get(t, 1.0))
            mv = float(mult_volatile.get(t, 1.0))
            m = p_t * mt + p_c * mc + p_v * mv
            eff.append(m)

        eff = np.array(eff, dtype=float)
        w = base * eff
        s = w.sum()
        if not np.isfinite(s) or s <= 1e-12:
            w = base.copy()
        else:
            w = w / s

        rows.append(pd.Series(w, index=alphas, name=dt))

    wdf = pd.DataFrame(rows).sort_index()
    wdf.index.name = "reb_date"   # ✅ 关键修复
    return wdf


# ---------------- portfolio construction (buffer + weighting + turnover control) ----------------
def build_weekly_weights_buffered(
    sig_long: pd.DataFrame,
    alpha_w_by_reb: pd.DataFrame,   # index=reb_date columns=alpha
    top_q: float,
    buffer_out_q: float,
    reb_freq: str,
    trade_lag_bdays: int,
    min_names: int,
    weighting: str = "score",       # equal|score|rank
) -> pd.DataFrame:
    """
    sig_long: columns [date,ticker,alpha,signal] (signal already direction-adjusted & clipped etc, NOT multiplied by alpha weights)
    Use reb_date bucketing: last signal within reb_date, then combine by alpha weights at reb_date.
    Then choose long/short with hysteresis buffer (in=top_q, out=buffer_out_q).
    """
    df = sig_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["alpha"] = df["alpha"].astype(str)
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "alpha", "signal"])
    if df.empty:
        return pd.DataFrame()

    # reb_date convention
    df["reb_date"] = df["date"].dt.to_period(reb_freq).dt.end_time
    df["reb_date"] = pd.to_datetime(df["reb_date"]).dt.normalize()
    if trade_lag_bdays and trade_lag_bdays > 0:
        df["reb_date"] = df["reb_date"] + BDay(int(trade_lag_bdays))

    # last obs per reb_date, ticker, alpha
    df = df.sort_values(["reb_date", "date"])
    df = df.groupby(["reb_date", "ticker", "alpha"], as_index=False).tail(1)
    if df.empty:
        return pd.DataFrame()

    # align reb_date with alpha_w_by_reb
    common_reb = df["reb_date"].unique()
    common_reb = pd.to_datetime(pd.Index(common_reb))
    common_reb = common_reb.intersection(alpha_w_by_reb.index)
    if len(common_reb) == 0:
        return pd.DataFrame()

    df = df[df["reb_date"].isin(common_reb)].copy()

    # combine into score per reb_date,ticker using alpha weights
    # merge weights
    #w_long = alpha_w_by_reb.loc[common_reb].reset_index().melt(id_vars="reb_date", var_name="alpha", value_name="w_alpha")
    tmp = alpha_w_by_reb.loc[common_reb].copy()

# ✅ 兜底：如果 index 没名字，就强制叫 reb_date
    if tmp.index.name is None or str(tmp.index.name).strip() == "":
        tmp.index.name = "reb_date"

    w_long = (
        tmp.reset_index()
            .melt(id_vars=tmp.index.name, var_name="alpha", value_name="w_alpha")
            .rename(columns={tmp.index.name: "reb_date"})
    )
    w_long["alpha"] = w_long["alpha"].astype(str)
    
    df = df.merge(w_long, on=["reb_date", "alpha"], how="left")
    df["w_alpha"] = df["w_alpha"].fillna(0.0)
    df["contrib"] = df["signal"] * df["w_alpha"]

    score = df.groupby(["reb_date", "ticker"], as_index=False)["contrib"].sum().rename(columns={"contrib": "score"})
    score = score.dropna(subset=["score"])
    if score.empty:
        return pd.DataFrame()

    top_q = float(top_q)
    if top_q <= 0 or top_q >= 0.5:
        top_q = 0.2

    buffer_out_q = float(buffer_out_q)
    if buffer_out_q <= top_q:
        buffer_out_q = min(0.45, top_q + 0.05)

    weighting = str(weighting).lower().strip()
    if weighting not in ("equal", "score", "rank"):
        weighting = "score"

    weights = []
    prev_long = set()
    prev_short = set()

    for dt, g in score.groupby("reb_date"):
        g = g.copy()
        if g["ticker"].nunique() < min_names:
            continue

        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        n = len(g)
        k_in = max(1, int(np.floor(n * top_q)))
        k_out = max(k_in, int(np.floor(n * buffer_out_q)))

        # ranks: 1..n
        g["rank"] = np.arange(1, n + 1)

        # long selection with buffer
        long_in = set(g.loc[g["rank"] <= k_in, "ticker"])
        long_out = set(g.loc[g["rank"] <= k_out, "ticker"])
        long_set = set(long_in)
        # keep previous longs if still within out threshold
        long_set |= (prev_long & long_out)

        # short selection with buffer (tail ranks)
        short_in = set(g.loc[g["rank"] >= (n - k_in + 1), "ticker"])
        short_out = set(g.loc[g["rank"] >= (n - k_out + 1), "ticker"])
        short_set = set(short_in)
        short_set |= (prev_short & short_out)

        # avoid overlap (rare but possible if k_out large)
        overlap = long_set & short_set
        if overlap:
            # remove weaker side by abs(score)
            tmp = g.set_index("ticker")["score"].to_dict()
            for t in list(overlap):
                if tmp.get(t, 0.0) >= 0:
                    short_set.discard(t)
                else:
                    long_set.discard(t)

        gg = g.set_index("ticker")

        # assign weights
        w = pd.Series(0.0, index=gg.index, dtype=float)

        if len(long_set) > 0:
            if weighting == "equal":
                w.loc[list(long_set)] = 1.0 / len(long_set)
            elif weighting == "rank":
                rr = gg.loc[list(long_set), "rank"].astype(float)
                # higher weight for better ranks
                s = (k_out + 1.0 - rr).clip(lower=0.0)
                if s.sum() <= 1e-12:
                    w.loc[list(long_set)] = 1.0 / len(long_set)
                else:
                    w.loc[list(long_set)] = (s / s.sum()).values
            else:  # score
                s = gg.loc[list(long_set), "score"].astype(float).clip(lower=0.0)
                if s.sum() <= 1e-12:
                    w.loc[list(long_set)] = 1.0 / len(long_set)
                else:
                    w.loc[list(long_set)] = (s / s.sum()).values

        if len(short_set) > 0:
            if weighting == "equal":
                w.loc[list(short_set)] = -1.0 / len(short_set)
            elif weighting == "rank":
                rr = gg.loc[list(short_set), "rank"].astype(float)
                # tail ranks: larger rank => stronger short
                s = (rr - (n - k_out)).clip(lower=0.0)
                if s.sum() <= 1e-12:
                    w.loc[list(short_set)] = -1.0 / len(short_set)
                else:
                    w.loc[list(short_set)] = -(s / s.sum()).values
            else:  # score
                s = (-gg.loc[list(short_set), "score"].astype(float)).clip(lower=0.0)
                if s.sum() <= 1e-12:
                    w.loc[list(short_set)] = -1.0 / len(short_set)
                else:
                    w.loc[list(short_set)] = -(s / s.sum()).values

        w = renormalize_long_short(w)

        weights.append(pd.Series(w.values, index=w.index, name=pd.to_datetime(dt)))

        prev_long = set(long_set)
        prev_short = set(short_set)

    if not weights:
        return pd.DataFrame()

    wdf = pd.DataFrame(weights).fillna(0.0).sort_index()
    return wdf


def apply_turnover_controller(
    w_target: pd.DataFrame,
    turnover_target: float = 0.30,
    turnover_band: float = 0.05,
    eta_init: float = 0.30,
    eta_min: float = 0.05,
    eta_max: float = 0.85,
    step_max: float = 0.05,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Adaptive smoothing:
      if turnover too high -> increase eta (more inertia)
      if turnover too low  -> decrease eta (follow target more)
    Also cap per-name change by step_max to avoid spikes.
    Return (weights, eta_series).
    """
    if w_target is None or w_target.empty:
        return w_target, pd.Series(dtype=float)

    tt = float(turnover_target)
    band = float(turnover_band)
    eta = float(eta_init)
    eta = min(max(eta, eta_min), eta_max)

    step_max = float(step_max)
    step_max = max(step_max, 0.0)

    cols = w_target.columns
    w_prev = None
    out = []
    etas = []

    for dt in w_target.index:
        wt = w_target.loc[dt].reindex(cols).fillna(0.0).astype(float)

        if w_prev is None:
            w = wt.copy()
            w = renormalize_long_short(w)
            out.append(pd.Series(w.values, index=cols, name=dt))
            etas.append(eta)
            w_prev = w.copy()
            continue

        # raw turnover to decide eta
        to_raw = 0.5 * (wt - w_prev).abs().sum()
        if np.isfinite(to_raw):
            if to_raw > tt + band:
                eta = min(eta_max, eta + 0.05)
            elif to_raw < max(0.0, tt - band):
                eta = max(eta_min, eta - 0.05)

        # smooth
        w = eta * w_prev + (1.0 - eta) * wt

        # per-name step limit
        if step_max > 0:
            delta = (w - w_prev).clip(-step_max, step_max)
            w = w_prev + delta

        w = renormalize_long_short(w)

        out.append(pd.Series(w.values, index=cols, name=dt))
        etas.append(eta)
        w_prev = w.copy()

    wdf = pd.DataFrame(out).fillna(0.0).sort_index()
    eta_s = pd.Series(etas, index=wdf.index, name="eta")
    return wdf, eta_s


def apply_vol_target(
    weights: pd.DataFrame,
    wk_ret_fwd: pd.DataFrame,
    target_ann_vol: float = 0.12,
    lookback_weeks: int = 26,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Vol target by scaling weights using realized portfolio vol (gross) lookback.
    Use shift(1) to avoid peeking.
    """
    if weights.empty:
        return weights, pd.Series(dtype=float)

    # compute gross with current weights (for realized vol estimate)
    common_dates = weights.index.intersection(wk_ret_fwd.index)
    if len(common_dates) < max(20, lookback_weeks):
        return weights, pd.Series(index=weights.index, data=np.nan, name="vol_scale")

    w = weights.loc[common_dates].copy()
    wk = wk_ret_fwd.loc[common_dates].reindex(columns=w.columns).fillna(0.0)
    gross = (w * wk).sum(axis=1).fillna(0.0)

    target_weekly = float(target_ann_vol) / np.sqrt(52.0)
    rv = gross.rolling(lookback_weeks, min_periods=max(10, lookback_weeks // 2)).std(ddof=1)

    scale = (target_weekly / rv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(scale_min, scale_max)
    scale = scale.shift(1)  # no peek
    scale = scale.reindex(weights.index).fillna(1.0)

    w2 = weights.mul(scale, axis=0).copy()
    # keep long/short normalized after scaling
    w2 = w2.apply(renormalize_long_short, axis=1)
    return w2, scale


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--panel", type=str, required=True)
    ap.add_argument("--seed_signals", type=str, required=True)
    ap.add_argument("--llm_dir", type=str, default="data/processed/signals_llm")
    ap.add_argument("--infer_json", type=str, required=True)
    ap.add_argument("--metrics_csv", type=str, default=None)

    ap.add_argument("--reb_freq", type=str, default="W-FRI")
    ap.add_argument("--top_q", type=float, default=0.2)
    ap.add_argument("--buffer_out_q", type=float, default=0.25)
    ap.add_argument("--trade_lag_bdays", type=int, default=1)
    ap.add_argument("--min_names", type=int, default=10)

    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)

    ap.add_argument("--winsorize", action="store_true")
    ap.add_argument("--winsor_lo", type=float, default=0.01)
    ap.add_argument("--winsor_hi", type=float, default=0.99)

    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--clip_z", type=float, default=None)

    # regime softmix
    ap.add_argument("--regime_temp", type=float, default=0.7)
    ap.add_argument("--regime_vol_window_w", type=int, default=13)
    ap.add_argument("--regime_trend_window_w", type=int, default=26)

    # alpha regime multipliers
    ap.add_argument("--mult_trend_mom", type=float, default=1.35)
    ap.add_argument("--mult_trend_rev", type=float, default=0.85)
    ap.add_argument("--mult_trend_vol", type=float, default=1.00)

    ap.add_argument("--mult_chop_mom", type=float, default=0.75)
    ap.add_argument("--mult_chop_rev", type=float, default=1.30)
    ap.add_argument("--mult_chop_vol", type=float, default=1.00)

    ap.add_argument("--mult_vol_mom", type=float, default=0.65)
    ap.add_argument("--mult_vol_rev", type=float, default=1.00)
    ap.add_argument("--mult_vol_vol", type=float, default=1.35)

    # weighting & turnover controller
    ap.add_argument("--weighting", type=str, default="score", help="equal|score|rank")
    ap.add_argument("--turnover_target", type=float, default=0.30)
    ap.add_argument("--turnover_band", type=float, default=0.05)
    ap.add_argument("--eta_init", type=float, default=0.30)
    ap.add_argument("--eta_min", type=float, default=0.05)
    ap.add_argument("--eta_max", type=float, default=0.85)
    ap.add_argument("--step_max", type=float, default=0.05)

    # optional vol target
    ap.add_argument("--vol_target_on", action="store_true")
    ap.add_argument("--vol_target_ann", type=float, default=0.12)
    ap.add_argument("--vol_lookback_weeks", type=int, default=26)
    ap.add_argument("--vol_scale_min", type=float, default=0.5)
    ap.add_argument("--vol_scale_max", type=float, default=1.5)

    ap.add_argument("--out_nav", type=str, default="results/portfolio_nav_llmweights_bayes_regime_buffer.csv")
    ap.add_argument("--out_summary", type=str, default="results/portfolio_summary_llmweights_bayes_regime_buffer.txt")
    ap.add_argument("--out_regime", type=str, default=None, help="optional: save regime probs csv")
    ap.add_argument("--out_eta", type=str, default=None, help="optional: save eta series csv")

    args = ap.parse_args()

    print("[INFO] Running backtest 09c (SoftRegime + Buffer + TurnoverCtrl + optional VolTarget)...")
    print(f"[INFO] python: {os.popen('python -V').read().strip()}")
    try_print_torch_info()

    for p in [args.panel, args.seed_signals, args.infer_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(os.path.dirname(args.out_nav) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    selected, w_alpha_base = load_infer_json(args.infer_json)
    direction = load_direction_from_metrics(args.metrics_csv, selected)

    print(f"[INFO] selected K={len(selected)}: {selected}")
    print(f"[INFO] base alpha weights: {w_alpha_base}")
    print(f"[INFO] direction(sign) from metrics: {direction}")

    # load panel
    panel = pd.read_parquet(args.panel)
    if "date" not in panel.columns or "ticker" not in panel.columns or "ret_1d" not in panel.columns:
        raise KeyError(f"panel must contain ['date','ticker','ret_1d']. Have={list(panel.columns)}")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    panel["ret_1d"] = pd.to_numeric(panel["ret_1d"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker", "ret_1d"])

    # regime probabilities per reb_date
    regime_probs = make_regime_probs(
    panel=panel,
    reb_freq=args.reb_freq,
    trade_lag_bdays=args.trade_lag_bdays,
    temp=max(0.8, float(args.regime_temp)),   # 建议先从 1.0 左右
)

    # alpha type multipliers
    mult_trending = {"mom": args.mult_trend_mom, "rev": args.mult_trend_rev, "vol": args.mult_trend_vol, "other": 1.0}
    mult_choppy = {"mom": args.mult_chop_mom, "rev": args.mult_chop_rev, "vol": args.mult_chop_vol, "other": 1.0}
    mult_volatile = {"mom": args.mult_vol_mom, "rev": args.mult_vol_rev, "vol": args.mult_vol_vol, "other": 1.0}

    alpha_w_by_reb = make_alpha_weights_by_regime_soft(
        selected=selected,
        base_w=w_alpha_base,
        regime_probs=regime_probs,
        mult_trending=mult_trending,
        mult_choppy=mult_choppy,
        mult_volatile=mult_volatile,
    )

    # load signals (seed + llm)
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

    # filter selected
    sig = sig[sig["alpha"].isin([str(a) for a in selected])].copy()
    if sig.empty:
        raise ValueError("No signals after filtering selected alphas. Check selected names vs signals alpha column.")

    # preprocess each alpha separately, keep signal (direction-adjusted) but NOT multiplied by weights yet
    parts = []
    for a in selected:
        sa = sig[sig["alpha"] == str(a)][["date", "ticker", "alpha", "signal"]].copy()
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

        sa["signal"] = sa["signal"] * float(direction.get(str(a), 1.0))
        parts.append(sa)

    if not parts:
        raise ValueError("All selected alphas had empty signals after processing.")

    sig_proc = pd.concat(parts, ignore_index=True)

    print(f"[INFO] sig_proc rows={len(sig_proc)} head:")
    print(sig_proc.head(3).to_string(index=False))

    # build buffered weekly weights (target)
    w_target = build_weekly_weights_buffered(
        sig_long=sig_proc,
        alpha_w_by_reb=alpha_w_by_reb,
        top_q=args.top_q,
        buffer_out_q=args.buffer_out_q,
        reb_freq=args.reb_freq,
        trade_lag_bdays=args.trade_lag_bdays,
        min_names=args.min_names,
        weighting=args.weighting,
    )
    if w_target.empty:
        raise ValueError("w_target empty. Check alignment of regime_probs/alpha_w_by_reb with signals reb_date.")

    # apply turnover controller (adaptive smoothing
        # apply turnover controller (adaptive smoothing)
    weights, eta_series = apply_turnover_controller(
        w_target=w_target,
        turnover_target=args.turnover_target,
        turnover_band=args.turnover_band,
        eta_init=args.eta_init,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        step_max=args.step_max,
    )

    # ---------- returns preparation ----------
    # daily -> weekly forward returns aligned with weights index
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()

    # 周收益 (W-FRI)
    week_ret = (1.0 + ret1).resample(args.reb_freq).prod() - 1.0
    week_ret_fwd = week_ret.shift(-1)

    # align to reb_date convention used in weights
    week_ret_fwd.index = pd.to_datetime(week_ret_fwd.index).normalize()
    if args.trade_lag_bdays and args.trade_lag_bdays > 0:
        week_ret_fwd.index = week_ret_fwd.index + BDay(int(args.trade_lag_bdays))

    # align dates
    common_dates = weights.index.intersection(week_ret_fwd.index)
    if len(common_dates) < 30:
        raise ValueError(f"Too few common dates after alignment: {len(common_dates)}")

    weights = weights.loc[common_dates].sort_index()
    #wk = week_ret_fwd.loc[common_dates].reindex(columns=weights.columns).fillna(0.0)
    wk_raw = week_ret_fwd.loc[common_dates].reindex(columns=weights.columns)

# 1) 把未来收益缺失的ticker的仓位置 0
    weights2 = weights.where(wk_raw.notna(), 0.0)

# 2) 逐行重新归一化 long/short（否则净暴露/毛暴露会漂）
    weights2 = weights2.apply(renormalize_long_short, axis=1)

# 3) 再把缺失收益填 0（因为对应仓位已为0）
    wk = wk_raw.fillna(0.0)

    gross = (weights2 * wk).sum(axis=1).dropna()

# turnover 也用 weights2（否则成本口径不一致）
    to = compute_turnover(weights2).reindex(gross.index).fillna(0.0)

    # ---------- optional vol target (after turnover control; scaling affects turnover less) ----------
    vol_scale = None
    if args.vol_target_on:
        weights, vol_scale = apply_vol_target(
            weights=weights,
            wk_ret_fwd=wk,
            target_ann_vol=args.vol_target_ann,
            lookback_weeks=args.vol_lookback_weeks,
            scale_min=args.vol_scale_min,
            scale_max=args.vol_scale_max,
        )
        # re-align again (apply_vol_target may reindex)
        weights = weights.loc[common_dates].sort_index()

    # ---------- portfolio returns ----------
    gross = (weights * wk).sum(axis=1).dropna()

    to = compute_turnover(weights).reindex(gross.index).fillna(0.0)
    cost = to * (args.cost_bps / 10000.0)
    slip = to * (args.slippage_bps / 10000.0) if args.slippage_bps and args.slippage_bps > 0 else 0.0
    net = gross - cost - slip

    nav_gross = (1.0 + gross.fillna(0.0)).cumprod()
    nav_net = (1.0 + net.fillna(0.0)).cumprod()

    # ---------- stats ----------
    ann_return_net = float(net.mean() * 52)
    ann_vol_net = float(net.std(ddof=1) * np.sqrt(52))
    v5, c5 = var_cvar(net, q=0.05)

    net_s = net.dropna()
    win_rate = float((net_s > 0).mean()) if len(net_s) else float("nan")

    avg_win = float(net_s[net_s > 0].mean()) if (net_s > 0).any() else float("nan")
    avg_loss = float(net_s[net_s < 0].mean()) if (net_s < 0).any() else float("nan")  # negative
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

    # regime counts (from soft probs -> argmax label)
    rp = regime_probs.loc[common_dates.intersection(regime_probs.index)].copy()
    if not rp.empty:
        lab = rp[["p_trending", "p_choppy", "p_volatile"]].idxmax(axis=1)
        lab = lab.map({"p_trending": "trending", "p_choppy": "choppy", "p_volatile": "volatile"})
        regime_counts = lab.value_counts().to_dict()
    else:
        regime_counts = {}

    # ---------- outputs ----------
    out = pd.DataFrame({
        "date": gross.index,
        "gross_ret": gross.values,
        "net_ret": net.values,
        "turnover": to.values,
        "nav_gross": nav_gross.values,
        "nav_net": nav_net.values,
    })
    out.to_csv(args.out_nav, index=False)

    # optional exports
    if args.out_regime:
        os.makedirs(os.path.dirname(args.out_regime) or ".", exist_ok=True)
        regime_probs.reset_index().rename(columns={"index": "reb_date"}).to_csv(args.out_regime, index=False)
        print(f"[OK] Saved regime probs: {args.out_regime}")

    if args.out_eta:
        os.makedirs(os.path.dirname(args.out_eta) or ".", exist_ok=True)
        eta_series.reindex(weights.index).rename("eta").reset_index().rename(columns={"index": "reb_date"}).to_csv(args.out_eta, index=False)
        print(f"[OK] Saved eta series: {args.out_eta}")

    # ---------- summary ----------
    summary = {
        "alphas_used": selected,
        "alpha_weights_base": w_alpha_base,
        "direction": direction,
        "rebalance": args.reb_freq,
        "TOP_Q": float(args.top_q),
        "BUFFER_OUT_Q": float(args.buffer_out_q),
        "TRADE_LAG_BDAYS": int(args.trade_lag_bdays),
        "MIN_NAMES": int(args.min_names),
        "WEIGHTING": str(args.weighting),

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
        "winsor_lo": float(args.winsor_lo),
        "winsor_hi": float(args.winsor_hi),
        "zscore": bool(args.zscore),
        "clip_z": None if args.clip_z is None else float(args.clip_z),

        # turnover control params
        "turnover_target": float(args.turnover_target),
        "turnover_band": float(args.turnover_band),
        "eta_init": float(args.eta_init),
        "eta_min": float(args.eta_min),
        "eta_max": float(args.eta_max),
        "step_max": float(args.step_max),

        # vol target
        "vol_target_on": bool(args.vol_target_on),
        "vol_target_ann": float(args.vol_target_ann),
        "vol_lookback_weeks": int(args.vol_lookback_weeks),
        "vol_scale_min": float(args.vol_scale_min),
        "vol_scale_max": float(args.vol_scale_max),

        "var_5pct_net": v5,
        "cvar_5pct_net": c5,
        "skew_net": float(net_s.skew()) if len(net_s) > 10 else float("nan"),
        "kurt_net": float(net_s.kurtosis()) if len(net_s) > 10 else float("nan"),

        # win metrics
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,

        "regime_counts": regime_counts,
    }

    with open(args.out_summary, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] Saved: {args.out_nav}")
    print(f"[OK] Saved: {args.out_summary}")
    print("[INFO] Summary (key):")
    for k in [
        "weeks", "sharpe_net", "ann_return_net", "ann_vol_net", "mdd_net",
        "avg_turnover", "win_rate", "profit_factor", "avg_win_loss_ratio",
        "max_win_streak", "max_loss_streak"
    ]:
        print(f"  {k}: {summary.get(k)}")
    print(f"  regime_counts: {summary.get('regime_counts')}")


if __name__ == "__main__":
    main()