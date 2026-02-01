#!/usr/bin/env python3
# 10_plot_nav_compare_3way.py
# Compare 3 strategies: no_llm vs llm vs llm_bayes_ema
# Robust alignment by mapping all dates to same weekly anchor (default W-FRI)

import os
import argparse
import numpy as np
import pandas as pd


def safe_read_nav(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} missing 'date' column. Have={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # derive nav_net if not exists
    if "nav_net" not in df.columns:
        if "net_ret" in df.columns:
            r = pd.to_numeric(df["net_ret"], errors="coerce").fillna(0.0)
            df["nav_net"] = (1.0 + r).cumprod()
        else:
            raise ValueError(f"{path} missing both 'nav_net' and 'net_ret'. Have={list(df.columns)}")

    # derive net_ret if not exists
    if "net_ret" not in df.columns:
        nav = pd.to_numeric(df["nav_net"], errors="coerce")
        df["net_ret"] = nav.pct_change().fillna(0.0)

    # turnover optional
    if "turnover" not in df.columns:
        df["turnover"] = np.nan

    df["nav_net"] = pd.to_numeric(df["nav_net"], errors="coerce")
    df["net_ret"] = pd.to_numeric(df["net_ret"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

    df = df.dropna(subset=["nav_net"])
    return df


def align_to_week(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """
    Robust: make sure 'date' column is unique, resample to weekly, and output
    columns: date, nav_net, net_ret, turnover
    """
    x = df.copy()

    # ---- normalize date column (guarantee ONE unique 'date') ----
    # If index is datetime-like, move it to a column called 'date'
    if "date" not in x.columns:
        if isinstance(x.index, pd.DatetimeIndex):
            x = x.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("Input df has no 'date' column and index is not DatetimeIndex")

    # If there are duplicated 'date' columns (this is your current crash), keep the first
    # pandas allows duplicate column labels; we must dedupe explicitly.
    if x.columns.duplicated().any():
        x = x.loc[:, ~x.columns.duplicated()].copy()

    # Now ensure 'date' is datetime
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date")

    # ---- pick available columns ----
    # your nav files should have these, but be defensive
    col_nav = "nav_net" if "nav_net" in x.columns else ("nav" if "nav" in x.columns else None)
    if col_nav is None:
        raise ValueError(f"Cannot find nav column in {list(x.columns)} (need nav_net or nav)")

    col_ret = "net_ret" if "net_ret" in x.columns else ("ret" if "ret" in x.columns else None)
    col_to  = "turnover" if "turnover" in x.columns else None

    # keep only relevant columns (avoid accidentally carrying a second date)
    keep = ["date", col_nav]
    if col_ret: keep.append(col_ret)
    if col_to:  keep.append(col_to)
    x = x[keep].copy()

    # ---- align to weekly anchor ----
    x = x.set_index("date").sort_index()

    # nav: use last value of week; returns: sum (or compound) depends on how you saved it
    # Here: if net_ret exists as weekly already, taking sum within week is OK for daily data;
    # but your inputs are weekly series already -> resample will be mostly identity.
    agg = {col_nav: "last"}
    if col_ret:
        agg[col_ret] = "sum"
    if col_to:
        agg[col_to] = "mean"

    out = x.resample(freq).agg(agg).dropna(subset=[col_nav]).reset_index()

    # rename to standard names
    out = out.rename(columns={col_nav: "nav_net"})
    if col_ret and col_ret != "net_ret":
        out = out.rename(columns={col_ret: "net_ret"})
    if col_to is None:
        out["turnover"] = np.nan

    # ensure final column order and uniqueness
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = out[["date", "nav_net", "net_ret", "turnover"]].sort_values("date").reset_index(drop=True)
    return out


def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) == 0:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def sharpe(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float(mu / sd * np.sqrt(ann_factor))


def ann_return(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.mean() * ann_factor)


def ann_vol(x: pd.Series, ann_factor=52) -> float:
    x = x.dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) * np.sqrt(ann_factor))


def win_rate(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    return float((x > 0).mean())


def profit_factor(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    pos = x[x > 0].sum()
    neg = x[x < 0].sum()
    if neg == 0:
        return float("nan")
    return float(pos / abs(neg))


def rolling_sharpe(x: pd.Series, window=52, ann_factor=52) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=1)
    rs = (mu / sd) * np.sqrt(ann_factor)
    rs = rs.replace([np.inf, -np.inf], np.nan)
    return rs


def plot_line(x, series_dict, title, ylabel, out_png):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    for name, y in series_dict.items():
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_llm", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--llm_bayes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="fig_3way")
    ap.add_argument("--roll_window", type=int, default=52)
    ap.add_argument("--align_freq", type=str, default="W-FRI", help="weekly anchor for alignment, default W-FRI")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw = {
        "no_llm": safe_read_nav(args.no_llm),
        "llm": safe_read_nav(args.llm),
        "llm_bayes_ema": safe_read_nav(args.llm_bayes),
    }

    data = {k: align_to_week(v, freq=args.align_freq) for k, v in raw.items()}

    # now inner join should work
    base = data["no_llm"][["date"]].copy()
    for k, df in data.items():
        base = base.merge(
            df.rename(columns={
                "nav_net": f"nav_net__{k}",
                "net_ret": f"net_ret__{k}",
                "turnover": f"turnover__{k}",
            }),
            on="date",
            how="inner"
        )

    if len(base) < 20:
        raise ValueError(f"Too few common dates after weekly alignment: {len(base)}. "
                         f"Try different --align_freq (e.g. W-THU) or check csv date ranges.")

    x = base["date"]

    nav_series = {k: base[f"nav_net__{k}"] for k in data.keys()}
    plot_line(x, nav_series, "NAV (net) - 3way", "nav_net",
              os.path.join(args.out_dir, f"{args.prefix}_nav_net.png"))

    dd_series = {}
    for k in data.keys():
        nav = base[f"nav_net__{k}"].astype(float)
        peak = nav.cummax()
        dd_series[k] = nav / peak - 1.0
    plot_line(x, dd_series, "Drawdown (net) - 3way", "drawdown",
              os.path.join(args.out_dir, f"{args.prefix}_dd_net.png"))

    rs_series = {}
    for k in data.keys():
        r = base[f"net_ret__{k}"].astype(float)
        rs_series[k] = rolling_sharpe(r, window=args.roll_window)
    plot_line(x, rs_series, f"Rolling Sharpe (net) - window={args.roll_window}w",
              "rolling_sharpe",
              os.path.join(args.out_dir, f"{args.prefix}_rollsharpe_net.png"))

    to_series = {k: base[f"turnover__{k}"] for k in data.keys()}
    plot_line(x, to_series, "Turnover - 3way", "turnover",
              os.path.join(args.out_dir, f"{args.prefix}_turnover.png"))

    rows = []
    for k in data.keys():
        r = base[f"net_ret__{k}"].astype(float)
        nav = base[f"nav_net__{k}"].astype(float)
        to = base[f"turnover__{k}"].astype(float)

        rows.append({
            "name": k,
            "weeks": int(r.dropna().shape[0]),
            "sharpe_net": sharpe(r),
            "ann_return_net": ann_return(r),
            "ann_vol_net": ann_vol(r),
            "mdd_net": max_drawdown(nav),
            "avg_turnover": float(to.dropna().mean()) if to.notna().any() else float("nan"),
            "win_rate": win_rate(r),
            "profit_factor": profit_factor(r),
            "final_nav_net": float(nav.dropna().iloc[-1]),
        })

    summary = pd.DataFrame(rows).sort_values("name")
    out_sum = os.path.join(args.out_dir, "summary_3way.csv")
    summary.to_csv(out_sum, index=False)

    print("[OK] Saved:")
    print(" ", os.path.join(args.out_dir, f"{args.prefix}_nav_net.png"))
    print(" ", os.path.join(args.out_dir, f"{args.prefix}_dd_net.png"))
    print(" ", os.path.join(args.out_dir, f"{args.prefix}_rollsharpe_net.png"))
    print(" ", os.path.join(args.out_dir, f"{args.prefix}_turnover.png"))
    print(" ", out_sum)


if __name__ == "__main__":
    main()