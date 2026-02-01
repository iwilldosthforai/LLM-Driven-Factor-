#!/usr/bin/env python3
# 10_plot_nav_compare_4way.py
# Plot NAV / Drawdown / Rolling Sharpe / Turnover for 4 strategies.

import os
import argparse
import numpy as np
import pandas as pd

def read_nav_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop duplicated columns (fix "date not unique")
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # common junk column
    if "Unnamed: 0" in df.columns and "date" not in df.columns:
        # sometimes the date got saved into Unnamed: 0
        df = df.rename(columns={"Unnamed: 0": "date"})

    # if still no date column, try index
    if "date" not in df.columns:
        # maybe first column is date-like
        if len(df.columns) > 0:
            c0 = df.columns[0]
            df = df.rename(columns={c0: "date"})

    if "date" not in df.columns:
        raise ValueError(f"[FATAL] {path}: cannot find 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")

    # drop duplicate dates (keep last)
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # choose nav/net_ret/turnover columns robustly
    # prefer net series
    nav_col = "nav_net" if "nav_net" in df.columns else ("nav_gross" if "nav_gross" in df.columns else None)
    ret_col = "net_ret" if "net_ret" in df.columns else ("gross_ret" if "gross_ret" in df.columns else None)

    if nav_col is None:
        # if only returns exist, rebuild nav
        if ret_col is None:
            raise ValueError(f"[FATAL] {path}: cannot find nav_* or *_ret columns")
        nav = (1.0 + pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)).cumprod()
        df["nav_net"] = nav
        nav_col = "nav_net"

    if ret_col is None:
        # if only nav exists, infer ret
        nav = pd.to_numeric(df[nav_col], errors="coerce")
        df["net_ret"] = nav.pct_change().fillna(0.0)
        ret_col = "net_ret"

    if "turnover" not in df.columns:
        df["turnover"] = 0.0

    out = df[["date", nav_col, ret_col, "turnover"]].copy()
    out = out.rename(columns={nav_col: "nav", ret_col: "ret"})
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce").fillna(0.0)
    out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce").fillna(0.0)

    # nav sanity
    out["nav"] = out["nav"].ffill()
    return out


def compute_drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.ffill()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return dd


def rolling_sharpe(r: pd.Series, window: int = 52, ann_factor: float = 52.0) -> pd.Series:
    r = r.fillna(0.0)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    out = (mu / sd) * np.sqrt(ann_factor)
    return out.replace([np.inf, -np.inf], np.nan)


def ensure_outdir(d: str):
    os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_llm", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--llm_bayes", required=True)
    ap.add_argument("--llm_bayes_regime", required=True)

    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--prefix", default="fig_4way")
    ap.add_argument("--roll_window", type=int, default=52)
    args = ap.parse_args()

    print("[INFO] Reading CSVs...")
    raw = {
        "no_llm": read_nav_csv(args.no_llm),
        "llm": read_nav_csv(args.llm),
        "llm_bayes": read_nav_csv(args.llm_bayes),
        "llm_bayes_regime": read_nav_csv(args.llm_bayes_regime),
    }

    # common start date = max(first date of each)
    starts = {k: v["date"].iloc[0] for k, v in raw.items() if len(v) > 0}
    start_date = max(starts.values())
    print("[INFO] Align start_date =", str(start_date.date()))

    # union index after start_date
    all_dates = sorted(set().union(*[
        set(v.loc[v["date"] >= start_date, "date"].tolist()) for v in raw.values()
    ]))
    if len(all_dates) < 20:
        raise ValueError(f"Too few dates after alignment: {len(all_dates)}")

    idx = pd.DatetimeIndex(all_dates, name="date")

    aligned = {}
    for k, df in raw.items():
        df2 = df[df["date"] >= start_date].copy().set_index("date").sort_index()
        # reindex to union, ffill nav, fill 0 ret/turnover where missing
        df2 = df2.reindex(idx)
        df2["nav"] = df2["nav"].ffill()
        df2["ret"] = df2["ret"].fillna(0.0)
        df2["turnover"] = df2["turnover"].fillna(0.0)
        # normalize NAV to 1 at start
        if pd.notna(df2["nav"].iloc[0]) and df2["nav"].iloc[0] != 0:
            df2["nav"] = df2["nav"] / df2["nav"].iloc[0]
        aligned[k] = df2

    # --- plotting ---
    import matplotlib.pyplot as plt

    ensure_outdir(args.out_dir)

    # 1) NAV
    plt.figure(figsize=(12, 6))
    for k, df in aligned.items():
        plt.plot(df.index, df["nav"], label=k)
    plt.title("NAV (normalized to 1 at common start)")
    plt.xlabel("date")
    plt.ylabel("nav")
    plt.legend()
    out1 = os.path.join(args.out_dir, f"{args.prefix}_nav.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    # 2) Drawdown
    plt.figure(figsize=(12, 6))
    for k, df in aligned.items():
        dd = compute_drawdown(df["nav"])
        plt.plot(df.index, dd, label=k)
    plt.title("Drawdown")
    plt.xlabel("date")
    plt.ylabel("drawdown")
    plt.legend()
    out2 = os.path.join(args.out_dir, f"{args.prefix}_drawdown.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    # 3) Rolling Sharpe
    plt.figure(figsize=(12, 6))
    for k, df in aligned.items():
        rs = rolling_sharpe(df["ret"], window=args.roll_window, ann_factor=52.0)
        plt.plot(df.index, rs, label=k)
    plt.title(f"Rolling Sharpe (window={args.roll_window} weeks)")
    plt.xlabel("date")
    plt.ylabel("rolling sharpe")
    plt.legend()
    out3 = os.path.join(args.out_dir, f"{args.prefix}_rollingsharpe.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    plt.close()

    # 4) Turnover (rolling mean)
    plt.figure(figsize=(12, 6))
    for k, df in aligned.items():
        t = df["turnover"].rolling(args.roll_window).mean()
        plt.plot(df.index, t, label=k)
    plt.title(f"Rolling Turnover Mean (window={args.roll_window} weeks)")
    plt.xlabel("date")
    plt.ylabel("turnover")
    plt.legend()
    out4 = os.path.join(args.out_dir, f"{args.prefix}_rollingturnover.png")
    plt.tight_layout()
    plt.savefig(out4, dpi=200)
    plt.close()

    # --- summary table ---
    rows = []
    for k, df in aligned.items():
        r = df["ret"]
        nav = df["nav"]
        ann_ret = r.mean() * 52.0
        ann_vol = r.std(ddof=1) * np.sqrt(52.0)
        shp = (ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan
        mdd = compute_drawdown(nav).min()
        win = (r > 0).mean()
        pf = (r[r > 0].sum() / abs(r[r < 0].sum())) if (r[r > 0].sum() > 0 and r[r < 0].sum() < 0) else np.nan
        rows.append({
            "name": k,
            "weeks": int(len(df)),
            "sharpe": float(shp),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "mdd": float(mdd),
            "avg_turnover": float(df["turnover"].mean()),
            "win_rate": float(win),
            "profit_factor": float(pf) if np.isfinite(pf) else np.nan,
        })

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out_csv = os.path.join(args.out_dir, f"summary_4way.csv")
    summary.to_csv(out_csv, index=False)

    print("[OK] Saved:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
    print(" ", out4)
    print(" ", out_csv)
    print("[INFO] Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()