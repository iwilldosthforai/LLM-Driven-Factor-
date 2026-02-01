#!/usr/bin/env python3
# 10_plot_nav_compare.py
# Compare NAV/Drawdown/Turnover/Rolling Sharpe between multiple backtest outputs.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NAV_CANDIDATES = ["nav_net", "nav", "nav_gross", "equity", "cum"]
RET_CANDIDATES = ["net_ret", "ret", "return", "pnl"]
TURN_CANDIDATES = ["turnover", "to", "turn"]


def _pick_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        # 有些人写成 Date / datetime
        for c in df.columns:
            if c.lower() in ["date", "datetime", "time"]:
                df = df.rename(columns={c: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"{path} missing date column. columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _compute_drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.astype(float)
    peak = nav.cummax()
    return nav / peak - 1.0


def _rolling_sharpe(ret: pd.Series, window=52, ann_factor=52) -> pd.Series:
    r = ret.astype(float)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    out = (mu / sd) * np.sqrt(ann_factor)
    return out.replace([np.inf, -np.inf], np.nan)


def _ensure_nav_and_ret(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    nav_col = _pick_col(df, NAV_CANDIDATES)
    ret_col = _pick_col(df, RET_CANDIDATES)
    turn_col = _pick_col(df, TURN_CANDIDATES)

    # 优先用 nav_net
    if nav_col is None:
        # 没 nav 就用 ret 造一个
        if ret_col is None:
            raise ValueError(f"Cannot find NAV or return columns. columns={list(df.columns)}")
        df["nav_net"] = (1.0 + df[ret_col].astype(float).fillna(0.0)).cumprod()
    else:
        # 统一名字叫 nav_net（方便后面画图）
        df["nav_net"] = df[nav_col].astype(float)

    # net_ret：如果没有就从 nav 推
    if ret_col is None:
        df["net_ret"] = df["nav_net"].pct_change().fillna(0.0)
    else:
        df["net_ret"] = df[ret_col].astype(float)

    # turnover 可选
    if turn_col is None:
        df["turnover"] = np.nan
    else:
        df["turnover"] = df[turn_col].astype(float)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--prefix", type=str, default="compare")
    ap.add_argument("--roll_weeks", type=int, default=52)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = args.inputs
    labels = args.labels
    if labels is None or len(labels) != len(paths):
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    series = []
    for p, lab in zip(paths, labels):
        df = _load_df(p)
        df = _ensure_nav_and_ret(df)
        df["dd_net"] = _compute_drawdown(df["nav_net"])
        df["roll_sharpe_net"] = _rolling_sharpe(df["net_ret"], window=args.roll_weeks, ann_factor=52)
        df["label"] = lab
        series.append(df)

    # date 对齐（交集）
    common = set(series[0]["date"])
    for df in series[1:]:
        common &= set(df["date"])
    common = sorted(common)
    if len(common) < 30:
        print(f"[WARN] common dates only {len(common)}; plotting anyway.")
    series = [df[df["date"].isin(common)].copy() for df in series]

    # 1) NAV
    plt.figure()
    for df in series:
        plt.plot(df["date"], df["nav_net"], label=df["label"].iloc[0])
    plt.title("NAV (net)")
    plt.xlabel("date"); plt.ylabel("nav_net")
    plt.legend()
    out1 = os.path.join(args.out_dir, f"fig_{args.prefix}_nav_net.png")
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.close()

    # 2) Drawdown
    plt.figure()
    for df in series:
        plt.plot(df["date"], df["dd_net"], label=df["label"].iloc[0])
    plt.title("Drawdown (net)")
    plt.xlabel("date"); plt.ylabel("drawdown")
    plt.legend()
    out2 = os.path.join(args.out_dir, f"fig_{args.prefix}_dd_net.png")
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()

    # 3) Rolling Sharpe
    plt.figure()
    for df in series:
        plt.plot(df["date"], df["roll_sharpe_net"], label=df["label"].iloc[0])
    plt.title(f"Rolling Sharpe (net) - {args.roll_weeks}w")
    plt.xlabel("date"); plt.ylabel("rolling_sharpe")
    plt.legend()
    out3 = os.path.join(args.out_dir, f"fig_{args.prefix}_rollsharpe_net.png")
    plt.tight_layout(); plt.savefig(out3, dpi=150); plt.close()

    # 4) Turnover（如果缺失会是空图/断线，但不报错）
    plt.figure()
    for df in series:
        plt.plot(df["date"], df["turnover"], label=df["label"].iloc[0])
    plt.title("Turnover")
    plt.xlabel("date"); plt.ylabel("turnover")
    plt.legend()
    out4 = os.path.join(args.out_dir, f"fig_{args.prefix}_turnover.png")
    plt.tight_layout(); plt.savefig(out4, dpi=150); plt.close()

    # summary csv
    rows = []
    for df in series:
        lab = df["label"].iloc[0]
        r = df["net_ret"].dropna()
        sd = r.std(ddof=1)
        shp = (r.mean() / sd * np.sqrt(52)) if len(r) > 10 and sd > 1e-12 else np.nan
        rows.append({
            "label": lab,
            "weeks": int(len(df)),
            "nav_net_end": float(df["nav_net"].iloc[-1]),
            "sharpe_net": float(shp) if np.isfinite(shp) else np.nan,
            "mdd_net": float(df["dd_net"].min()),
            "avg_turnover": float(np.nanmean(df["turnover"].values)),
        })
    summ = pd.DataFrame(rows)
    out5 = os.path.join(args.out_dir, f"summary_{args.prefix}.csv")
    summ.to_csv(out5, index=False)

    print("[OK] Saved:")
    for p in [out1, out2, out3, out4, out5]:
        print(" ", p)


if __name__ == "__main__":
    main()