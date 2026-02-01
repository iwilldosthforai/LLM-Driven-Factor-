import os
import json
import ast
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Defaults
# =========================
DEFAULT_IN_PANEL = "data/processed/panel.parquet"
DEFAULT_CANDIDATES = "results/llm_candidates.jsonl"
DEFAULT_OUT_DIR = "data/processed/signals_llm"  # directory (hive partitioned: alpha=...)


# =========================
# Factor operators
# =========================
def cs_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank in [0,1] each date. index must be MultiIndex (date,ticker)."""
    return s.groupby(level=0).rank(pct=True)

def zscore_cs(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score each date."""
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

def sign(x: pd.Series) -> pd.Series:
    return np.sign(x)

def abs_(x: pd.Series) -> pd.Series:
    return np.abs(x)


# =========================
# Safety: AST validator
# =========================
ALLOWED_VARS = {"ret1", "close", "high", "low", "vol", "dollar_vol"}

ALLOWED_FUNCS = {
    "ts_sum", "ts_mean", "ts_std", "ts_max", "ts_min",
    "cs_rank", "zscore_cs",
    "safe_log", "abs", "sign"
}

ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

class SafeExprValidator(ast.NodeVisitor):
    def __init__(self):
        self.errors: List[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def visit_Expression(self, node: ast.Expression):
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in ALLOWED_VARS and node.id not in ALLOWED_FUNCS:
            self.error(f"Disallowed name: {node.id}")

    def visit_Call(self, node: ast.Call):
        # only simple calls: func(...)
        if not isinstance(node.func, ast.Name):
            self.error("Only simple function calls are allowed.")
        else:
            if node.func.id not in ALLOWED_FUNCS:
                self.error(f"Disallowed function: {node.func.id}")
        if node.keywords:
            self.error("Keyword args are not allowed.")
        for a in node.args:
            self.visit(a)

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, ALLOWED_BINOPS):
            self.error(f"Disallowed binary op: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, ALLOWED_UNARYOPS):
            self.error(f"Disallowed unary op: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, (int, float)):
            self.error("Only int/float constants are allowed.")

    def generic_visit(self, node):
        allowed = (
            ast.Expression, ast.Name, ast.Call, ast.BinOp, ast.UnaryOp,
            ast.Constant, ast.Load
        )
        if not isinstance(node, allowed):
            self.error(f"Disallowed syntax: {type(node).__name__}")
            return
        super().generic_visit(node)

def validate_expr(expr: str) -> Tuple[bool, List[str]]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]
    v = SafeExprValidator()
    v.visit(tree)
    return (len(v.errors) == 0), v.errors

def safe_eval_expr(expr: str, env: Dict[str, Any]) -> pd.Series:
    tree = ast.parse(expr, mode="eval")
    code = compile(tree, "<alpha_expr>", "eval")
    return eval(code, {"__builtins__": {}}, env)


# =========================
# IO helpers
# =========================
def load_candidates(path: str, max_n: int = 0) -> List[Dict[str, Any]]:
    """Read JSONL. Each line is {"name":..., "expr":..., "tag":...}."""
    out = []
    if not os.path.exists(path):
        print(f"[WARN] candidates file not found: {path}")
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            name = obj.get("name") or obj.get("alpha")
            expr = obj.get("expr")
            if not name or not expr:
                continue
            out.append({"name": str(name), "expr": str(expr), "tag": obj.get("tag", "")})
            if max_n > 0 and len(out) >= max_n:
                break
    return out

def make_long(sig: pd.Series) -> pd.DataFrame:
    """
    Convert MultiIndex(date,ticker) Series -> DataFrame(date,ticker,signal)
    IMPORTANT: do NOT include 'alpha' column here because we use hive partition alpha=...
    """
    df = sig.rename("signal").reset_index()
    # reset_index yields columns ['date','ticker','signal'] if MultiIndex names exist,
    # but be robust to default names.
    cols = list(df.columns)
    if len(cols) >= 3:
        # attempt to enforce names
        df = df.rename(columns={cols[0]: "date", cols[1]: "ticker", cols[2]: "signal"})
    return df[["date", "ticker", "signal"]]

def write_partition(df: pd.DataFrame, out_dir: str, alpha: str):
    """
    Write parquet under out_dir/alpha=<alpha>/part_xxx.parquet
    The alpha value will be available as a hive partition column when reading the directory.
    """
    part_dir = os.path.join(out_dir, f"alpha={alpha}")
    os.makedirs(part_dir, exist_ok=True)
    fn = os.path.join(part_dir, f"part_{pd.Timestamp.now().value}.parquet")
    df.to_parquet(fn, index=False)

def clean_out_dir(out_dir: str, force: bool):
    if os.path.exists(out_dir) and force:
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_panel", default=DEFAULT_IN_PANEL)
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max_alphas", type=int, default=0, help="0 means all")
    ap.add_argument("--dropna", action="store_true")
    ap.add_argument("--clip", type=float, default=0.0, help="clip abs(signal) to this value; 0 disables")
    ap.add_argument("--force_clean", action="store_true", help="delete out_dir before writing")
    args = ap.parse_args()

    clean_out_dir(args.out_dir, args.force_clean)

    # Load panel
    panel = pd.read_parquet(args.in_panel)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Build MultiIndex
    idx = pd.MultiIndex.from_frame(panel[["date", "ticker"]])
    idx = idx.set_names(["date", "ticker"])

    # Column mapping
    if "adj_close" in panel.columns:
        close_col = "adj_close"
    elif "close" in panel.columns:
        close_col = "close"
    else:
        raise ValueError("panel must contain 'adj_close' or 'close'")

    required = {
        close_col: "close",
        "high": "high",
        "low": "low",
        "volume": "vol",
        "ret_1d": "ret1",
        "dollar_vol": "dollar_vol",
    }
    for c in required.keys():
        if c not in panel.columns:
            raise ValueError(f"panel missing required column: {c}")

    # Base series
    close = pd.Series(panel[close_col].to_numpy(), index=idx)
    high = pd.Series(panel["high"].to_numpy(), index=idx)
    low = pd.Series(panel["low"].to_numpy(), index=idx)
    vol = pd.Series(panel["volume"].to_numpy(), index=idx)
    ret1 = pd.Series(panel["ret_1d"].to_numpy(), index=idx).fillna(0.0)
    dollar_vol = pd.Series(panel["dollar_vol"].to_numpy(), index=idx)

    # Safe env
    env: Dict[str, Any] = {
        "ret1": ret1,
        "close": close,
        "high": high,
        "low": low,
        "vol": vol,
        "dollar_vol": dollar_vol,
        "ts_sum": ts_sum,
        "ts_mean": ts_mean,
        "ts_std": ts_std,
        "ts_max": ts_max,
        "ts_min": ts_min,
        "cs_rank": cs_rank,
        "zscore_cs": zscore_cs,
        "safe_log": safe_log,
        "abs": abs_,
        "sign": sign,
    }

    # Load candidates
    cands = load_candidates(args.candidates, max_n=args.max_alphas)
    print(f"Loaded candidates: {len(cands)} from {args.candidates}")
    if len(cands) == 0:
        print("[HINT] If this shows 0, your working directory/path is wrong. "
              "From /Users/xsy/Desktop/data run: `ls results/llm_candidates.jsonl`.")
        return

    ok_cnt, fail_cnt, dup_cnt = 0, 0, 0
    seen_expr = set()

    for i, cand in enumerate(cands, 1):
        name = cand["name"]
        expr = cand["expr"].strip()

        if expr in seen_expr:
            dup_cnt += 1
            continue
        seen_expr.add(expr)

        valid, errs = validate_expr(expr)
        if not valid:
            fail_cnt += 1
            print(f"[{i:03d}] SKIP {name}: invalid expr -> {errs[:2]}")
            continue

        try:
            sig = safe_eval_expr(expr, env)
            if not isinstance(sig, pd.Series):
                raise TypeError("Expression did not produce a pandas Series.")

            sig = sig.replace([np.inf, -np.inf], np.nan)

            if args.clip and args.clip > 0:
                sig = sig.clip(lower=-args.clip, upper=args.clip)

            if args.dropna:
                sig = sig.dropna()

            df = make_long(sig)
            df = df.dropna(subset=["signal"])

            if len(df) == 0:
                fail_cnt += 1
                print(f"[{i:03d}] SKIP {name}: empty after dropna")
                continue

            write_partition(df, args.out_dir, name)
            ok_cnt += 1

            if i % 10 == 0:
                print(f"Progress: {i}/{len(cands)} | ok={ok_cnt} fail={fail_cnt} dup={dup_cnt}")

        except Exception as e:
            fail_cnt += 1
            print(f"[{i:03d}] FAIL {name}: {type(e).__name__}: {e}")

    print(f"Done. ok={ok_cnt}, fail={fail_cnt}, dup_expr={dup_cnt}")
    print(f"Saved partitioned parquet dataset at: {args.out_dir}")
    print("Tip: pd.read_parquet(out_dir) will load all partitions (and provide alpha as a partition column).")


if __name__ == "__main__":
    main()