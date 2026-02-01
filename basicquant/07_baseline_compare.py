import os
import json
import numpy as np
import pandas as pd

PANEL_PATH   = "data/processed/panel.parquet"
SIGNALS_PATH = "data/processed/signals.parquet"
SEL_PATH     = "results/selected_alphas.json"

TOP_Q = 0.2
COST_BPS = 10
REB_FREQ = "W-FRI"

def build_weekly_weights(score_long: pd.DataFrame, top_q=0.2) -> pd.DataFrame:
    df = score_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","ticker"])
    df["reb_date"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    df = df.sort_values(["reb_date","ticker","date"]).groupby(["reb_date","ticker"]).tail(1)
    df["rank"] = df.groupby("reb_date")["score"].rank(method="average", pct=True)

    long_mask  = df["rank"] >= (1 - top_q)
    short_mask = df["rank"] <= top_q

    df["w"] = 0.0
    df.loc[long_mask, "w"] =  1.0
    df.loc[short_mask,"w"] = -1.0

    def norm_grp(g):
        w = g["w"].to_numpy()
        pos = (w > 0).sum()
        neg = (w < 0).sum()
        if pos > 0:
            w[w > 0] = w[w > 0] / pos
        if neg > 0:
            w[w < 0] = w[w < 0] / neg
        g["w"] = w
        return g

    df = df.groupby("reb_date", group_keys=False).apply(norm_grp)
    w = df.pivot(index="reb_date", columns="ticker", values="w").fillna(0.0)
    w.index.name = "date"
    return w

def turnover(weights: pd.DataFrame) -> pd.Series:
    return 0.5 * weights.diff().abs().sum(axis=1)

def sharpe(x: pd.Series, ann=52) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(ann))

def mdd(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())

def run_strategy(score_long: pd.DataFrame, week_ret: pd.DataFrame, name: str) -> dict:
    weights = build_weekly_weights(score_long, TOP_Q)
    common_dates = weights.index.intersection(week_ret.index)
    weights = weights.loc[common_dates]
    wk = week_ret.loc[common_dates, weights.columns]

    gross = (weights * wk).sum(axis=1)
    to = turnover(weights).reindex(gross.index).fillna(0.0)
    net = gross - to * (COST_BPS/10000.0)

    nav = (1+net.fillna(0.0)).cumprod()
    return {
        "name": name,
        "weeks": int(len(net)),
        "sharpe_net": sharpe(net),
        "mdd_net": mdd(nav),
        "avg_turnover": float(to.mean()),
        "final_nav": float(nav.iloc[-1]),
    }

def main():
    # 周收益
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    ret1 = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    week_ret = (1.0 + ret1).resample(REB_FREQ).prod() - 1.0

    sig = pd.read_parquet(SIGNALS_PATH)
    sig["date"] = pd.to_datetime(sig["date"])

    with open(SEL_PATH, "r") as f:
        sel = json.load(f)
    selected = [x["alpha"] for x in sel["selected"]]

    # Ours: 选中因子平均
    ours = sig[sig["alpha"].isin(selected)].groupby(["date","ticker"], as_index=False)["signal"].mean()
    ours = ours.rename(columns={"signal":"score"})
    res_ours = run_strategy(ours, week_ret, "ours_8_alphas")

    # Baseline: 单因子 mom_20_rank
    base_alpha = "mom_20_rank"
    base = sig[sig["alpha"]==base_alpha][["date","ticker","signal"]].rename(columns={"signal":"score"})
    res_base = run_strategy(base, week_ret, f"baseline_{base_alpha}")

    out = pd.DataFrame([res_base, res_ours])
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/baseline_compare.csv", index=False)
    print(out)

if __name__ == "__main__":
    main()