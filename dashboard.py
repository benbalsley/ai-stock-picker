import glob
import pandas as pd
import numpy as np


def summarize_backtest_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] reading {path}: {e}")
        return None

    if df.empty:
        return None

    pnl = df["pnl_dollars"]
    pnl_pct = df["pnl_pct"]

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    summary = {
        "file": path,
        "trades": len(df),
        "win_rate": len(wins) / len(df) * 100.0,
        "total_pnl": pnl.sum(),
        "total_return_pct": pnl_pct.sum(),
        "avg_win": wins.mean() if len(wins) else np.nan,
        "avg_loss": losses.mean() if len(losses) else np.nan,
    }
    return summary


def main():
    files = sorted(glob.glob("backtest_results_*.csv"))
    if not files:
        print("[INFO] No backtest_results_*.csv files found.")
        return

    summaries = []
    for f in files:
        s = summarize_backtest_csv(f)
        if s:
            summaries.append(s)

    if not summaries:
        print("[INFO] No valid summaries.")
        return

    df = pd.DataFrame(summaries)
    print("\n========== BACKTEST DASHBOARD ==========")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=========================================")


if __name__ == "__main__":
    main()

