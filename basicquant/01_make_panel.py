import os
import pandas as pd

IN_PATH = "data/raw/us_ohlcv_yf.parquet"     # 你刚下载好的文件
OUT_PATH = "data/processed/panel.parquet"    # 我们要生成的面板

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = pd.read_parquet(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # 1) 当日收益（用 adj_close）
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()

    # 2) 未来 5 日收益（label）
    df["fwd_ret_5d"] = df.groupby("ticker")["adj_close"].shift(-5) / df["adj_close"] - 1.0

    # 3) （可选）美元成交额，后面做流动性/成本很有用
    df["dollar_vol"] = df["adj_close"] * df["volume"]

    # 基础清理
    df = df.dropna(subset=["adj_close", "volume"])
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} | rows={len(df):,} | tickers={df['ticker'].nunique()}")

if __name__ == "__main__":
    main()