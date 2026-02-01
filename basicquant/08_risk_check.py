import pandas as pd
import numpy as np

NAV_PATH = "results/portfolio_nav.csv"
PANEL_PATH = "data/processed/panel.parquet"

def ols_beta(y, x):
    # y = a + b*x
    x = x.dropna()
    y = y.reindex(x.index).dropna()
    x = x.reindex(y.index)
    if len(y) < 30:
        return np.nan, np.nan
    X = np.vstack([np.ones(len(x)), x.values]).T
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    resid = y.values - X @ b
    r2 = 1 - (resid.var() / (y.values.var() + 1e-12))
    return float(b[1]), float(r2)

def main():
    nav = pd.read_csv(NAV_PATH)
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date").sort_index()
    strat = nav["net_ret"]

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    # 等权市场日收益
    mkt_day = ret1.mean(axis=1)
    # 聚合到周（W-FRI 标记的收益通常代表“上一周->本周五”的区间）
    mkt_week = (1.0 + mkt_day).resample("W-FRI").prod() - 1.0

    # 与策略对齐：策略 net_ret 在日期 t 表示“t -> 下一周五”的收益（backtest 使用了 week_ret.shift(-1)）
    # 因此市场周收益也要 shift(-1) 才是同一持有期
    mkt_week = mkt_week.shift(-1)

    # 对齐到策略周序列
    mkt_week = mkt_week.reindex(strat.index).dropna()
    strat = strat.reindex(mkt_week.index).dropna()

    beta, r2 = ols_beta(strat, mkt_week)

    print("Risk check (weekly):")
    print("  corr(strat, mkt):", float(strat.corr(mkt_week)))
    print("  beta to mkt:", beta)
    print("  R^2:", r2)
    print("  strat mean/std:", float(strat.mean()), float(strat.std(ddof=1)))
    print("  mkt   mean/std:", float(mkt_week.mean()), float(mkt_week.std(ddof=1)))

if __name__ == "__main__":
    main()