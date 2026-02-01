import os
import numpy as np
import pandas as pd

IN_PATH  = "data/processed/panel.parquet"
OUT_PATH = "data/processed/signals.parquet"

# ---------- helpers ----------
def cs_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank in [0,1] each date."""
    return s.groupby(level=0).rank(pct=True)

def zscore_cs(s: pd.Series) -> pd.Series:
    """Cross-sectional zscore each date."""
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / (g.transform("std") + 1e-12)

def ts_mean(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1).rolling(w).mean().reset_index(level=0, drop=True)

def ts_std(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1).rolling(w).std().reset_index(level=0, drop=True)

def ts_sum(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1).rolling(w).sum().reset_index(level=0, drop=True)

def ts_max(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1).rolling(w).max().reset_index(level=0, drop=True)

def ts_min(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level=1).rolling(w).min().reset_index(level=0, drop=True)

def safe_log(x: pd.Series) -> pd.Series:
    return np.log(np.clip(x, 1e-12, None))

def make_long(sig: pd.Series, alpha: str) -> pd.DataFrame:
    df = sig.rename("signal").reset_index()
    df["alpha"] = alpha
    return df[["date", "ticker", "alpha", "signal"]]

# ---------- main ----------
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    panel = pd.read_parquet(IN_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"])

    # 用 MultiIndex 方便截面/时序运算：level0=date, level1=ticker
    idx = pd.MultiIndex.from_frame(panel[["date", "ticker"]])
    close = pd.Series(panel["adj_close"].to_numpy(), index=idx)
    high  = pd.Series(panel["high"].to_numpy(),      index=idx)
    low   = pd.Series(panel["low"].to_numpy(),       index=idx)
    vol   = pd.Series(panel["volume"].to_numpy(),    index=idx)
    ret1  = pd.Series(panel["ret_1d"].to_numpy(),    index=idx)
    dollar_vol = pd.Series(panel["dollar_vol"].to_numpy(), index=idx)

    # 缺失收益先填 0（第一天/停牌等）
    ret1 = ret1.fillna(0.0)

    signals = []

    # --- Momentum / Reversal ---
    mom_20 = ts_sum(ret1, 20)
    signals.append(make_long(cs_rank(mom_20), "mom_20_rank"))

    mom_60 = ts_sum(ret1, 60)
    signals.append(make_long(cs_rank(mom_60), "mom_60_rank"))

    rev_5 = -ts_sum(ret1, 5)
    signals.append(make_long(cs_rank(rev_5), "rev_5_rank"))

    rev_10 = -ts_sum(ret1, 10)
    signals.append(make_long(cs_rank(rev_10), "rev_10_rank"))

    # --- Volatility / Risk ---
    vol_20 = ts_std(ret1, 20)
    signals.append(make_long(cs_rank(-vol_20), "lowvol_20_rank"))  # 低波更优

    tr = (high - low) / (close.replace(0, np.nan))
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr_14 = ts_mean(tr, 14)
    signals.append(make_long(cs_rank(-atr_14), "lowatr_14_rank"))

    down_vol_20 = ts_std(np.minimum(ret1, 0.0), 20)
    signals.append(make_long(cs_rank(-down_vol_20), "low_downvol_20_rank"))

    # --- Breakout / Position ---
    max_20 = ts_max(close, 20)
    breakout_20 = (close / max_20) - 1.0
    signals.append(make_long(cs_rank(breakout_20), "breakout_20_rank"))

    min_20 = ts_min(close, 20)
    position_20 = (close - min_20) / (max_20 - min_20 + 1e-12)  # [0,1]
    signals.append(make_long(position_20, "position_20"))  # 已经是0~1，无需rank

    # --- Volume / Price-Volume ---
    vol_chg_5 = vol.groupby(level=1).pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signals.append(make_long(cs_rank(vol_chg_5), "volchg_5_rank"))

    # 成交量相对强度：vol / ts_mean(vol,20)
    vol_rel_20 = vol / (ts_mean(vol, 20) + 1e-12)
    signals.append(make_long(cs_rank(vol_rel_20), "volrel_20_rank"))

    # OBV (简化)
    direction = np.sign(ret1)
    obv = (direction * vol).groupby(level=1).cumsum()
    obv_20 = obv - obv.groupby(level=1).shift(20)
    signals.append(make_long(cs_rank(obv_20), "obv_20_rank"))

    # --- Liquidity proxy (Amihud-like) ---
    illiq = (np.abs(ret1) / (dollar_vol + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    illiq_20 = ts_mean(illiq, 20)
    signals.append(make_long(cs_rank(-illiq_20), "liquidity_20_rank"))  # 流动性更好(illiq更小)

    # --- Mean reversion vs trend strength ---
    # 趋势强度：过去20天上涨天数比例
    up = (ret1 > 0).astype(float)
    up_ratio_20 = ts_mean(up, 20)
    signals.append(make_long(up_ratio_20, "up_ratio_20"))

    # 价的偏离：close / ts_mean(close,20) - 1
    dev_20 = close / (ts_mean(close, 20) + 1e-12) - 1.0
    signals.append(make_long(cs_rank(-np.abs(dev_20)), "mean_revert_20_rank"))

    out = pd.concat(signals, ignore_index=True)

    # 清理极端/缺失
    out["signal"] = out["signal"].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["signal"])
    out.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH} | rows={len(out):,} | alphas={out['alpha'].nunique()} | tickers={out['ticker'].nunique()}")

if __name__ == "__main__":
    main()