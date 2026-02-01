import pandas as pd

PATH = "results/walkforward_oos_nav.csv"

def main():
    df = pd.read_csv(PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["nav"] = (1 + df["net_ret"].fillna(0)).cumprod()
    dd = df["nav"] / df["nav"].cummax() - 1

    # 最大回撤点
    i = dd.idxmin()
    print("Worst drawdown:")
    print(df.loc[i, ["date","net_ret","nav"]])
    print("DD:", dd.loc[i])

    # 输出崩盘前后 20 周
    lo = max(0, i-20)
    hi = min(len(df)-1, i+20)
    print("\nWindow around crash:")
    print(df.loc[lo:hi, ["date","net_ret","nav"]].to_string(index=False))

if __name__ == "__main__":
    main()