import os
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

OUT_PATH = "data/raw/us_ohlcv_yf.parquet"

# 先用一组大盘股/高流动性股票跑通（后面你再扩展到 100/500/全市场）
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V",
    "LLY","AVGO","UNH","XOM","MA","COST","PG","JNJ","HD","BAC",
    "ABBV","KO","PEP","WMT","ADBE","CRM","NFLX","AMD","ORCL","INTC",
    "CSCO","QCOM","TMO","ACN","MCD","DIS","NKE","PFE","CVX","MRK"
]

START = "2010-01-01"
END   = "2025-01-01"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance 有时会返回 MultiIndex 列（tuple），先扁平化，再统一成 snake_case 小写。
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 常见形态：('Open','AAPL') -> 取第 0 层：Open
        df.columns = df.columns.get_level_values(0)

    # 统一成字符串 + 小写 + 空格转下划线
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def download_one(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # ✅ 关键：先处理 MultiIndex/tuple 列名
    df = _normalize_columns(df)

    df = df.reset_index()
    df = _normalize_columns(df)

    df["ticker"] = ticker

    # 兼容不同版本可能出现的字段名差异
    # 目标字段：date, open, high, low, close, adj_close, volume
    rename_map = {}
    if "adj_close" not in df.columns:
        # 有些版本/场景可能是 adjclose
        if "adjclose" in df.columns:
            rename_map["adjclose"] = "adj_close"
        # 极端情况下可能是 "adj_close" 以外的形式（保底）
        elif "adj_close" in df.columns:
            pass

    if rename_map:
        df = df.rename(columns=rename_map)

    needed = ["date","ticker","open","high","low","close","adj_close","volume"]

    # 如果缺列，直接返回空表，让上层逻辑跳过（避免 KeyError）
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    return df[needed]

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    frames = []
    for t in tqdm(TICKERS, desc="Downloading"):
        # 简单重试，避免偶发网络/限流
        for attempt in range(3):
            try:
                df = download_one(t)
                if not df.empty:
                    frames.append(df)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[FAIL] {t}: {e}")
                else:
                    time.sleep(2 * (attempt + 1))

    if not frames:
        raise RuntimeError("No data downloaded. Check network or tickers.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["ticker","date"])
    # 写 parquet 需要 pyarrow 或 fastparquet；如果没装则自动降级成 csv
    try:
        all_df.to_parquet(OUT_PATH, index=False)
        saved_path = OUT_PATH
    except ImportError as e:
        csv_path = os.path.splitext(OUT_PATH)[0] + ".csv"
        all_df.to_csv(csv_path, index=False)
        saved_path = csv_path
        print("[WARN] Parquet engine missing (pyarrow/fastparquet). Saved CSV instead:", csv_path)

    print(
        f"Saved: {saved_path} | rows={len(all_df):,} | "
        f"tickers={all_df['ticker'].nunique()}"
    )

if __name__ == "__main__":
    main()