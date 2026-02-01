import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_PATH = "results/walkforward_oos_nav.csv"
DYN_PATH  = "results/walkforward_oos_nav_dynamic.csv"
OUT_PNG1  = "results/fig_oos_nav_compare.png"
OUT_PNG2  = "results/fig_oos_dd_compare.png"

def load_nav(path, ret_col="net_ret"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if "nav" in df.columns:
        nav = df.set_index("date")["nav"].astype(float)
    else:
        r = df[ret_col].fillna(0.0).astype(float)
        nav = (1 + r).cumprod()
        nav.index = df["date"].values
        nav = pd.Series(nav.values, index=pd.to_datetime(df["date"]))
    return nav

def drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0

def main():
    os.makedirs("results", exist_ok=True)

    nav_b = load_nav(BASE_PATH)
    nav_d = load_nav(DYN_PATH)

    # 对齐日期（用交集）
    idx = nav_b.index.intersection(nav_d.index)
    nav_b = nav_b.reindex(idx)
    nav_d = nav_d.reindex(idx)

    # 1) NAV 对比（用对数轴更直观）
    plt.figure()
    plt.plot(nav_b.index, nav_b.values, label="baseline (static)")
    plt.plot(nav_d.index, nav_d.values, label="dynamic weighting")
    plt.yscale("log")
    plt.title("Walk-forward OOS NAV (log scale)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG1, dpi=200)
    plt.close()

    # 2) Drawdown 对比
    dd_b = drawdown(nav_b)
    dd_d = drawdown(nav_d)

    plt.figure()
    plt.plot(dd_b.index, dd_b.values, label="baseline (static)")
    plt.plot(dd_d.index, dd_d.values, label="dynamic weighting")
    plt.title("Walk-forward OOS Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG2, dpi=200)
    plt.close()

    print("Saved:")
    print(" ", OUT_PNG1)
    print(" ", OUT_PNG2)

if __name__ == "__main__":
    main()