#!/usr/bin/env python3
# 11_plot_nav_compare_5way.py
# Compare 5 methods: NAV / Drawdown / Turnover / Rolling Sharpe / Rolling Vol + summary csv

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- helpers ----------------------
def _fix_duplicate_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    # If columns contain duplicates like ['date','date',...], keep the first one.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def read_nav_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _fix_duplicate_date_cols(df)

    if "date" not in df.columns:
        # Sometimes index-like col exists
        for c in df.columns:
            if str(c).lower() in ("datetime", "time", "dt"):
                df = df.rename(columns={c: "date"})
                break

    if "date" not in df.columns:
        raise ValueError(f"[{path}] missing 'date' column. columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # normalize columns
    col_map = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in col_map:
                return col_map[c.lower()]
        return None

    nav = pick("nav_net", "nav", "cum_net", "equity", "portfolio_nav", "nav_value")
    ret = pick("net_ret", "ret", "return", "pnl", "weekly_ret")
    to  = pick("turnover", "to", "avg_turnover")

    if nav is None:
        # if only returns exist, build nav
        if ret is None:
            raise ValueError(f"[{path}] missing nav and net_ret columns. columns={list(df.columns)}")
        r = pd.to_numeric(df[ret], errors="coerce").fillna(0.0)
        nav_s = (1.0 + r).cumprod()
        df["nav_net"] = nav_s
        nav = "nav_net"

    df["nav_net"] = pd.to_numeric(df[nav], errors="coerce")
    if ret is None:
        # derive returns from nav
        df["net_ret"] = df["nav_net"].pct_change().fillna(0.0)
    else:
        df["net_ret"] = pd.to_numeric(df[ret], errors="coerce").fillna(0.0)

    if to is None:
        df["turnover"] = np.nan
    else:
        df["turnover"] = pd.to_numeric(df[to], errors="coerce")

    keep = df[["date", "nav_net", "net_ret", "turnover"]].copy()
    return keep


def align_weekly(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date")
    df = df.set_index("date")

    # NAV: take last of week
    nav = df["nav_net"].resample(freq).last()

    # Return: compound within week
    ret = (1.0 + df["net_ret"]).resample(freq).prod() - 1.0

    # Turnover: mean within week (if exists)
    if df["turnover"].notna().any():
        to = df["turnover"].resample(freq).mean()
    else:
        to = pd.Series(index=nav.index, data=np.nan)

    out = pd.DataFrame({"nav_net": nav, "net_ret": ret, "turnover": to})
    out = out.dropna(subset=["nav_net"])
    out.index = pd.to_datetime(out.index).normalize()
    return out


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def sharpe(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(mu / sd * np.sqrt(ann_factor))


def rolling_sharpe(x: pd.Series, window: int = 52, ann_factor=52) -> pd.Series:
    x = x.fillna(0.0)
    mu = x.rolling(window, min_periods=max(10, window // 2)).mean()
    sd = x.rolling(window, min_periods=max(10, window // 2)).std(ddof=1)
    rs = (mu / sd.replace(0, np.nan)) * np.sqrt(ann_factor)
    return rs


def rolling_vol(x: pd.Series, window: int = 52, ann_factor=52) -> pd.Series:
    x = x.fillna(0.0)
    sd = x.rolling(window, min_periods=max(10, window // 2)).std(ddof=1)
    return sd * np.sqrt(ann_factor)


def save_lineplot(x, ys: dict, title: str, ylabel: str, outpath: str):
    plt.figure(figsize=(12, 6))
    for name, y in ys.items():
        plt.plot(x, y, label=name, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True, help="Method1 CSV (no LLM)")
    ap.add_argument("--m2", required=True, help="Method2 CSV (LLM)")
    ap.add_argument("--m3", required=True, help="Method3 CSV (LLM+Bayes+EMA)")
    ap.add_argument("--m4", required=True, help="Method4 CSV (LLM+Bayes+Regime)")
    ap.add_argument("--m5", required=True, help="Method5 CSV (LLM+Bayes+Regime+Buffer+TurnoverCtrl)")
    ap.add_argument("--names", default="no_llm,llm,llm_bayes_ema,regime,regime_buffer_toctrl")
    ap.add_argument("--align_freq", default="W-FRI")
    ap.add_argument("--roll_window", type=int, default=52)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="fig_5way")
    args = ap.parse_args()

    paths = [args.m1, args.m2, args.m3, args.m4, args.m5]
    names = [s.strip() for s in args.names.split(",")]
    if len(names) != 5:
        names = ["m1", "m2", "m3", "m4", "m5"]

    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(args.out_dir, exist_ok=True)

    raw = {}
    for name, p in zip(names, paths):
        df = read_nav_csv(p)
        wk = align_weekly(df, freq=args.align_freq)
        raw[name] = wk

    # intersection on weekly index
    idx = None
    for name, df in raw.items():
        idx = df.index if idx is None else idx.intersection(df.index)

    if idx is None or len(idx) < 30:
        # Print ranges to help debug alignment
        print("[FATAL] Too few common dates after alignment:", 0 if idx is None else len(idx))
        for name, df in raw.items():
            if df.empty:
                print(f"  {name}: EMPTY after align_weekly")
            else:
                print(f"  {name}: {df.index.min().date()} -> {df.index.max().date()}  (n={len(df)})")
        raise ValueError(f"Too few common dates after alignment: {0 if idx is None else len(idx)}")

    data = {k: v.loc[idx].copy() for k, v in raw.items()}

    # Normalize NAV to 1 at start
    navs = {}
    dds = {}
    turns = {}
    rs = {}
    rv = {}

    for name, df in data.items():
        nav = df["nav_net"].astype(float)
        nav = nav / float(nav.iloc[0]) if float(nav.iloc[0]) != 0 else nav
        navs[name] = nav

        peak = nav.cummax()
        dds[name] = nav / peak - 1.0

        turns[name] = df["turnover"].astype(float)

        r = df["net_ret"].astype(float)
        rs[name] = rolling_sharpe(r, window=args.roll_window)
        rv[name] = rolling_vol(r, window=args.roll_window)

    # ---- plots ----
    prefix = args.prefix
    out_dir = args.out_dir

    save_lineplot(idx, navs, "NAV (Normalized) - 5 Methods", "NAV", os.path.join(out_dir, f"{prefix}_nav.png"))
    save_lineplot(idx, dds,  "Drawdown - 5 Methods", "Drawdown", os.path.join(out_dir, f"{prefix}_drawdown.png"))
    save_lineplot(idx, turns, "Turnover (Weekly Avg) - 5 Methods", "Turnover", os.path.join(out_dir, f"{prefix}_turnover.png"))
    save_lineplot(idx, rs,   f"Rolling Sharpe (window={args.roll_window}) - 5 Methods", "Rolling Sharpe", os.path.join(out_dir, f"{prefix}_rolling_sharpe.png"))
    save_lineplot(idx, rv,   f"Rolling Vol (ann, window={args.roll_window}) - 5 Methods", "Rolling Vol", os.path.join(out_dir, f"{prefix}_rolling_vol.png"))

    # ---- summary ----
    rows = []
    for name, df in data.items():
        r = df["net_ret"].astype(float).fillna(0.0)
        nav = (navs[name]).copy()
        to = df["turnover"].astype(float)

        rows.append({
            "method": name,
            "weeks": int(len(r)),
            "sharpe": sharpe(r),
            "ann_return": float(r.mean() * 52),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(52)),
            "mdd": max_drawdown(nav),
            "avg_turnover": float(to.mean()) if to.notna().any() else np.nan,
            "win_rate": float((r > 0).mean()),
        })

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    sum_path = os.path.join(out_dir, f"summary_5way.csv")
    summary.to_csv(sum_path, index=False)

    print("[OK] Saved figures to:", out_dir)
    print("  -", f"{prefix}_nav.png")
    print("  -", f"{prefix}_drawdown.png")
    print("  -", f"{prefix}_turnover.png")
    print("  -", f"{prefix}_rolling_sharpe.png")
    print("  -", f"{prefix}_rolling_vol.png")
    print("[OK] Saved summary:", sum_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()