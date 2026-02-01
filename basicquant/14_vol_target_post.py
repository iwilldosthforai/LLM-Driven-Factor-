import numpy as np
import pandas as pd

IN_PATH = "results/walkforward_oos_nav.csv"
OUT_PATH = "results/walkforward_oos_nav_voltarget.csv"

LOOKBACK_W = 26
TARGET_WEEKLY_VOL = 0.10 / np.sqrt(52)   # 目标年化10%
MAX_LEV = 3.0

def main():
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    r = df["net_ret"].fillna(0.0)

    # 实现波动率估计（滚动 std）
    vol = r.rolling(LOOKBACK_W).std(ddof=1)
    lev = (TARGET_WEEKLY_VOL / (vol + 1e-12)).clip(lower=0.0, upper=MAX_LEV)
    # 前 LOOKBACK_W 没 vol，用 1
    lev = lev.fillna(1.0)

    r2 = r * lev
    df["lev"] = lev
    df["net_ret_vt"] = r2
    df["nav_vt"] = (1 + r2).cumprod()

    # 指标
    ann = 52
    mu = r2.mean()
    sd = r2.std(ddof=1)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(ann)
    peak = df["nav_vt"].cummax()
    mdd = (df["nav_vt"] / peak - 1).min()

    df.to_csv(OUT_PATH, index=False)
    print("Vol-targeted OOS:")
    print("  sharpe:", float(sharpe))
    print("  mdd:", float(mdd))
    print("  final_nav:", float(df["nav_vt"].iloc[-1]))

if __name__ == "__main__":
    main()