import pandas as pd
import numpy as np

NAV_PATH = "results/portfolio_nav.csv"

def main():
    df = pd.read_csv(NAV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 基本统计
    print("Weekly net return stats:")
    print(df["net_ret"].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]))

    # 最差的 10 周（检查有没有极端 -100% 或异常）
    worst = df.nsmallest(10, "net_ret")[["date","net_ret","gross_ret","turnover","nav_net"]]
    print("\nWorst 10 weeks (net_ret):")
    print(worst.to_string(index=False))

    # 最好的 10 周（看有没有极端大于 50%/100%）
    best = df.nlargest(10, "net_ret")[["date","net_ret","gross_ret","turnover","nav_net"]]
    print("\nBest 10 weeks (net_ret):")
    print(best.to_string(index=False))

    # 检查是否有离谱值
    print("\nChecks:")
    print("Any net_ret <= -1 ?", (df["net_ret"] <= -1).any())
    print("Any net_ret >=  1 ?", (df["net_ret"] >=  1).any())
    print("Max |net_ret|:", float(df["net_ret"].abs().max()))

if __name__ == "__main__":
    main()