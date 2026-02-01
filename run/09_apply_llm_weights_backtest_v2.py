#!/usr/bin/env python3
# 09_apply_llm_weights_backtest.py
# Apply LLM/Bayes-inferred alpha weights and run portfolio backtest + optional EMA smoothing on portfolio weights.

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


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


def smooth_weights_ema(weights: pd.DataFrame, eta: float) -> pd.DataFrame:
    """
    EMA smoothing on portfolio weights across rebalance dates.
    w_t = eta * w_{t-1} + (1-eta) * w_target_t
    keep row-wise sum unchanged (no need; it's long-short sums to ~0 anyway), just smooth elementwise.
    """
    if weights is None or weights.empty:
        return weights

    e = float(eta)
    e = min(max(e, 0.0), 0.999)

    w = weights.fillna(0.0).astype(float).copy()
    out = w.copy()

    prev = None
    for i, dt in enumerate(w.index):
        if prev is None:
            out.loc[dt] = w.loc[dt]
        else:
            out.loc[dt] = e * prev + (1.0 - e) * w.loc[dt]
        prev = out.loc[dt].values.astype(float, copy=True)

    return out


def build_weekly_weights_from_score(
    score_long: pd.DataFrame,
    top_q: float = 0.2,
    reb_freq: str = "W-FRI",
    trade_lag_bdays: int = 1,
    winsorize: bool = False,
    winsor_lo: float = 0.01,
    winsor_hi: float = 0.99,
    zscore: bool = False,
    clip_z: float = 0.0,
    long_short: bool = True,
    min_names: int = 10,
) -> pd.DataFrame:
    """
    Input: long df columns ['date','ticker','score'] (or MultiIndex date,ticker + score)
    Output: weights wide df index=reb_date, columns=ticker, values=weight
    Always returns DataFrame (may be empty), never None.
    """
    if score_long is None:
        return pd.DataFrame()

    df = score_long
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({"score": df})

    if ("date" not in df.columns) or ("ticker" not in df.columns):
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
            df = df.copy()
            df["date"] = df.index.get_level_values(0)
            df["ticker"] = df.index.get_level_values(1)
        else:
            return pd.DataFrame()

    if "score" not in df.columns:
        if df.shape[1] >= 1:
            df = df.rename(columns={df.columns[0]: "score"})
        else:
            return pd.DataFrame()

    df = df[["date", "ticker", "score"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "ticker", "score"])
    if df.empty:
        return pd.DataFrame()

    def _winsorize_one_day(g: pd.DataFrame) -> pd.DataFrame:
        s = g["score"].astype(float)
        lo = s.quantile(winsor_lo)
        hi = s.quantile(winsor_hi)
        g = g.copy()
        g["score"] = s.clip(lo, hi)
        return g

    def _zscore_one_day(g: pd.DataFrame) -> pd.DataFrame:
        s = g["score"].astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        g = g.copy()
        g["score"] = 0.0 if sd <= 1e-12 else (s - mu) / sd
        return g

    if winsorize:
        df = df.groupby("date", group_keys=False).apply(_winsorize_one_day)

    if zscore:
        df = df.groupby("date", group_keys=False).apply(_zscore_one_day)

    if clip_z and clip_z > 0:
        df["score"] = df["score"].clip(-float(clip_z), float(clip_z))

    df["reb_date"] = df["date"].dt.to_period(reb_freq).dt.end_time
    df["reb_date"] = pd.to_datetime(df["reb_date"]).dt.normalize()

    if trade_lag_bdays and trade_lag_bdays > 0:
        df["reb_date"] = df["reb_date"] + pd.offsets.BDay(int(trade_lag_bdays))

    df = df.sort_values(["reb_date", "date"])
    df = df.groupby(["reb_date", "ticker"], as_index=False).tail(1)
    if df.empty:
        return pd.DataFrame()

    top_q = float(top_q)
    if top_q <= 0 or top_q >= 0.5:
        top_q = 0.2

    def _make_weights_one_reb(g: pd.DataFrame) -> pd.DataFrame:
        reb = getattr(g, "name", None)
        if reb is None:
            if "reb_date" in g.columns and len(g) > 0:
                reb = g["reb_date"].iloc[0]
            else:
                return pd.DataFrame(columns=["reb_date", "ticker", "w"])

        g = g.copy()
        g = g.dropna(subset=["score"])
        if g.empty or g["ticker"].nunique() < min_names:
            return pd.DataFrame(columns=["reb_date", "ticker", "w"])

        g = g.sort_values("score", ascending=False)
        n = len(g)
        k = max(1, int(np.floor(n * top_q)))

        if long_short:
            long_leg = g.head(k).copy()
            short_leg = g.tail(k).copy()
            long_leg["w"] =  1.0 / max(1, len(long_leg))
            short_leg["w"] = -1.0 / max(1, len(short_leg))
            out = pd.concat([long_leg[["ticker", "w"]], short_leg[["ticker", "w"]]], ignore_index=True)
        else:
            long_leg = g.head(k).copy()
            long_leg["w"] = 1.0 / max(1, len(long_leg))
            out = long_leg[["ticker", "w"]].copy()

        out.insert(0, "reb_date", pd.to_datetime(reb))
        return out[["reb_date", "ticker", "w"]]

    weights_long = df.groupby("reb_date", group_keys=False).apply(_make_weights_one_reb)
    if weights_long is None or len(weights_long) == 0:
        return pd.DataFrame()

    weights = (
        weights_long.pivot_table(index="reb_date", columns="ticker", values="w", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
        .astype(float)
    )
    return weights


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, required=True)
    ap.add_argument("--seed_signals", type=str, required=True)
    ap.add_argument("--llm_dir", type=str, default="data/processed/signals_llm")
    ap.add_argument("--infer_json", type=str, required=True)
    ap.add_argument("--metrics_csv", type=str, default=None)

    ap.add_argument("--reb_freq", type=str, default="W-FRI")
    ap.add_argument("--top_q", type=float, default=0.2)
    ap.add_argument("--trade_lag_bdays", type=int, default=1)

    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)

    ap.add_argument("--winsorize", action="store_true")
    ap.add_argument("--winsor_lo", type=float, default=0.01)
    ap.add_argument("--winsor_hi", type=float, default=0.99)

    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--clip_z", type=float, default=None)

    # ✅ NEW: EMA smoothing
    ap.add_argument("--ema_eta", type=float, default=None,
                    help="EMA smoothing on portfolio weights across rebalance dates. 0~1. None=off")

    ap.add_argument("--out_nav", type=str, default="results/portfolio_nav_llm.csv")
    ap.add_argument("--out_summary", type=str, default="results/portfolio_summary_llm.txt")
    args = ap.parse_args()

    print("[INFO] Running backtest 09...")
    print(f"[INFO] python: {os.popen('python -V').read().strip()}")
    try_print_torch_info()

    print(f"[INFO] panel: {args.panel}")
    print(f"[INFO] seed_signals: {args.seed_signals}")
    print(f"[INFO] llm_dir: {args.llm_dir}")
    print(f"[INFO] infer_json: {args.infer_json}")
    print(f"[INFO] metrics_csv: {args.metrics_csv}")

    if not os.path.exists(args.panel):
        raise FileNotFoundError(args.panel)
    if not os.path.exists(args.seed_signals):
        raise FileNotFoundError(args.seed_signals)
    if not os.path.exists(args.infer_json):
        raise FileNotFoundError(args.infer_json)

    os.makedirs(os.path.dirname(args.out_nav) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    selected, w_alpha = load_infer_json(args.infer_json)
    print(f"[INFO] selected K={len(selected)}: {selected}")
    print(f"[INFO] weights: {w_alpha}")

    direction = load_direction_from_metrics(args.metrics_csv, selected)
    print(f"[INFO] direction(sign) from metrics: {direction}")

    if args.clip_z is not None:
        print(f"[INFO] clip_z enabled: {args.clip_z}")
    if args.ema_eta is not None:
        print(f"[INFO] ema_eta enabled: {args.ema_eta}")

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
    if len(sig) == 0:
        raise ValueError("No signals after filtering selected alphas. Check selected names vs signals alpha column.")

    # ---------- per-alpha transform + combine ----------
    parts = []
    for a in selected:
        sa = sig[sig["alpha"] == a][["date", "ticker", "signal"]].copy()
        if len(sa) == 0:
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
    score = score.rename(columns={"signal": "score"})

    print(f"[INFO] score_long columns: {list(score.columns)} rows={len(score)}")
    print("[INFO] score_long head:")
    print(score.head(3).to_string(index=False))

    # ---------- build weekly weights ----------
    weights = build_weekly_weights_from_score(
        score_long=score,
        top_q=args.top_q,
        trade_lag_bdays=args.trade_lag_bdays,
        reb_freq=args.reb_freq,
    )

    # ✅ EMA smoothing on portfolio weights
    if args.ema_eta is not None:
        weights = smooth_weights_ema(weights, eta=float(args.ema_eta))

        # ---------- returns ----------
    panel = pd.read_parquet(args.panel)
    if "date" not in panel.columns or "ticker" not in panel.columns or "ret_1d" not in panel.columns:
        raise KeyError(f"panel must contain ['date','ticker','ret_1d']. Have={list(panel.columns)}")

    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    panel["ret_1d"] = pd.to_numeric(panel["ret_1d"], errors="coerce")

    # --- weekly returns (match weight reb_date convention) ---
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()

    # 周收益：以 W-FRI 对齐；week_ret 的 index 是 period end（周五）
    week_ret = (1.0 + ret1).resample(args.reb_freq).prod() - 1.0

    # weights 用“本周信号 -> 下周收益”逻辑，所以 forward 一周
    week_ret_fwd = week_ret.shift(-1)

    # ✅ 关键：week_ret_fwd 的 index 必须和 weights.index 一样的 reb_date 口径
    week_ret_fwd.index = pd.to_datetime(week_ret_fwd.index).normalize()
    if args.trade_lag_bdays and args.trade_lag_bdays > 0:
        week_ret_fwd.index = week_ret_fwd.index + BDay(int(args.trade_lag_bdays))

    # --- align dates and tickers ---
    common_dates = weights.index.intersection(week_ret_fwd.index)
    weights = weights.loc[common_dates].sort_index()

    # 用 reindex 保证列对齐（缺的ticker补0）
    wk = week_ret_fwd.loc[common_dates].reindex(columns=weights.columns)

    # --- portfolio returns ---
    gross = (weights * wk).sum(axis=1).dropna()

    to = compute_turnover(weights).reindex(gross.index).fillna(0.0)
    cost = to * (args.cost_bps / 10000.0)
    slip = to * (args.slippage_bps / 10000.0) if args.slippage_bps and args.slippage_bps > 0 else 0.0
    net = gross - cost - slip

    nav_gross = (1.0 + gross.fillna(0.0)).cumprod()
    nav_net = (1.0 + net.fillna(0.0)).cumprod()

    ann_return_net = float(net.mean() * 52)
    ann_vol_net = float(net.std(ddof=1) * np.sqrt(52))
    v5, c5 = var_cvar(net, q=0.05)

    # --- win-rate & streaks ---
    net_s = net.dropna()
    win_rate = float((net_s > 0).mean()) if len(net_s) else float("nan")

    avg_win = float(net_s[net_s > 0].mean()) if (net_s > 0).any() else float("nan")
    avg_loss = float(net_s[net_s < 0].mean()) if (net_s < 0).any() else float("nan")  # 负数
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

    # ---------- summary ----------
    summary = {
        "alphas_used": selected,
        "alpha_weights": w_alpha,
        "direction": direction,
        "TOP_Q": float(args.top_q),
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
        "winsor_lo": float(args.winsor_lo),
        "winsor_hi": float(args.winsor_hi),
        "zscore": bool(args.zscore),
        "clip_z": None if args.clip_z is None else float(args.clip_z),
        "ema_eta": None if args.ema_eta is None else float(args.ema_eta),
        "var_5pct_net": v5,
        "cvar_5pct_net": c5,
        "skew_net": float(net_s.skew()) if len(net_s) > 10 else float("nan"),
        "kurt_net": float(net_s.kurtosis()) if len(net_s) > 10 else float("nan"),

        # ✅ win metrics
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }

    out = pd.DataFrame({
        "date": gross.index,
        "gross_ret": gross.values,
        "net_ret": net.values,
        "turnover": to.values,
        "nav_gross": nav_gross.values,
        "nav_net": nav_net.values,
    })
    out.to_csv(args.out_nav, index=False)

    with open(args.out_summary, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] Saved: {args.out_nav}")
    print(f"[OK] Saved: {args.out_summary}")
    print("[INFO] Summary (key):")
    for k in ["weeks", "sharpe_net", "ann_return_net", "ann_vol_net", "mdd_net", "avg_turnover",
              "win_rate", "profit_factor", "avg_win_loss_ratio", "max_win_streak", "max_loss_streak"]:
        print(f"  {k}: {summary.get(k)}")



if __name__ == "__main__":
    main()