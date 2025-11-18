import datetime as dt
import math
import csv

import numpy as np
import pandas as pd
import yfinance as yf

from config import ACCOUNT_EQUITY, RISK_PER_TRADE_PCT
from ai_stock_picker_v2 import (
    load_universe,
    fetch,
    score_ticker_v2,
    get_market_regime,
)


############################################################
# Helpers
############################################################

def download_history_for_universe(tickers, start, end):
    """
    Download daily history for all tickers once.
    Returns dict: {ticker: DataFrame}
    """
    data = {}
    for t in tickers:
        try:
            df = yf.download(
                t,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False
            )
            if df.empty:
                print(f"[WARN] No data for {t}")
                continue
            df = df.dropna()
            data[t] = df
            print(f"[DATA] {t}: {len(df)} rows")
        except Exception as e:
            print(f"[ERROR] downloading {t}: {e}")
    return data


def get_spy_trading_days(start, end):
    """
    Use SPY daily data as the trading calendar (simple & robust).
    """
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Could not get SPY history to build trading calendar.")
    return list(df.index.date)


############################################################
# Trade simulation (same as V1 backtest)
############################################################

def build_position_size(signal):
    """Risk-based sizing logic: use ATR and account size."""
    price = signal["price"]
    atr = signal["atr"]
    risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE_PCT

    stop_distance = 1.5 * atr
    if stop_distance <= 0:
        return None

    raw_qty = risk_dollars / stop_distance
    qty = max(1, math.floor(raw_qty))

    max_affordable_qty = math.floor(ACCOUNT_EQUITY / price)
    if max_affordable_qty <= 0:
        return None

    qty = min(qty, max_affordable_qty)
    return qty


def simulate_trade_for_day(signal_date, signal, df, trading_days):
    """
    For a given signal generated on 'signal_date':
    - Enter at NEXT trading day's open
    - Exit at SAME day close
    - Return PnL and record
    """
    try:
        idx = trading_days.index(signal_date)
    except ValueError:
        return None  # date not in calendar

    if idx + 1 >= len(trading_days):
        # No next day data
        return None

    trade_date = trading_days[idx + 1]  # we trade on next session

    df_trade = df[df.index.date == trade_date]
    if df_trade.empty:
        return None

    open_px = float(df_trade["Open"].iloc[0])
    close_px = float(df_trade["Close"].iloc[0])

    qty = build_position_size(signal)
    if qty is None:
        return None

    # V2 is long-only
    direction = 1

    pnl_per_share = (close_px - open_px) * direction
    pnl_dollars = pnl_per_share * qty
    pnl_pct = pnl_dollars / ACCOUNT_EQUITY * 100.0

    record = {
        "signal_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "ticker": signal["ticker"],
        "trend": "long",
        "entry_price": round(open_px, 4),
        "exit_price": round(close_px, 4),
        "qty": qty,
        "pnl_dollars": round(pnl_dollars, 2),
        "pnl_pct": round(pnl_pct, 4),
        "gap_pct": round(signal["gap_pct"] * 100.0, 2),
        "atr": round(signal["atr"], 4),
        "rel_vol": round(signal["rel_vol"], 4),
        "vol_score": round(signal["vol_score"], 4),
        "score": round(signal["score"], 4),
    }
    return record


############################################################
# Backtest driver for V2
############################################################

def run_backtest_v2(days_back=252, output_file="backtest_results_v2.csv"):
    """
    Backtest roughly 1 year of V2 signals:
    - One signal per trading day (best candidate that passes V2 filters)
    - Market regime filter (SPY/QQQ) applied per day
    - Enter next day open, exit same day close
    """
    today = dt.date.today()

    history_buffer_days = 90
    start_date = today - dt.timedelta(days=days_back + history_buffer_days)
    end_date = today

    print(f"[INFO] V2 Backtest window (signals): last {days_back} trading days")
    print(f"[INFO] History pulled from {start_date} to {end_date}")

    trading_days = get_spy_trading_days(start_date, end_date)
    if len(trading_days) < days_back + 10:
        print("[WARN] Fewer trading days than expected; check SPY data.")

    if len(trading_days) <= days_back + 1:
        signal_days = trading_days[:-1]
    else:
        signal_days = trading_days[-(days_back + 1):-1]

    universe = load_universe()
    if not universe:
        print("[ERROR] Universe is empty. Aborting V2 backtest.")
        return

    print(f"[INFO] Universe size: {len(universe)} tickers")

    all_data = download_history_for_universe(universe, start_date, end_date)

    fieldnames = [
        "signal_date",
        "trade_date",
        "ticker",
        "trend",
        "entry_price",
        "exit_price",
        "qty",
        "pnl_dollars",
        "pnl_pct",
        "gap_pct",
        "atr",
        "rel_vol",
        "vol_score",
        "score",
    ]
    records = []

    for d in signal_days:
        print(f"\n[DAY] {d} – computing V2 signal...")

        # Market regime filter for that day
        regime = get_market_regime(d)
        if not regime.get("allow_longs", True):
            print("[DAY] Regime says no longs today; skipping.")
            continue

        best_signal = None

        for t, df in all_data.items():
            try:
                info = score_ticker_v2(t, d, df)
            except Exception as e:
                print(f"[WARN] scoring error for {t} on {d}: {e}")
                continue
            if info is None:
                continue

            if (best_signal is None) or (info["score"] > best_signal["score"]):
                best_signal = info

        if best_signal is None:
            print(f"[DAY] {d}: no V2 candidate passed filters.")
            continue

        df_t = all_data[best_signal["ticker"]]
        rec = simulate_trade_for_day(d, best_signal, df_t, trading_days)
        if rec is None:
            print(f"[DAY] {d}: could not simulate trade for {best_signal['ticker']}")
            continue

        print(
            f"[TRADE] {rec['trade_date']} {rec['ticker']} LONG "
            f"entry={rec['entry_price']} exit={rec['exit_price']} "
            f"PnL=${rec['pnl_dollars']} ({rec['pnl_pct']}%)"
        )

        records.append(rec)

    if records:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"\n[RESULT] V2 backtest written to {output_file}")
    else:
        print("\n[RESULT] No V2 trades generated. Nothing written.")
        return

    pnl_series = [r["pnl_dollars"] for r in records]
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p <= 0]

    total_pnl = round(sum(pnl_series), 2)
    total_return_pct = round(sum(r["pnl_pct"] for r in records), 2)
    win_rate = round(len(wins) / len(records) * 100.0, 2) if records else 0.0
    avg_win = round(np.mean(wins), 2) if wins else 0.0
    avg_loss = round(np.mean(losses), 2) if losses else 0.0

    print("\n========== V2 BACKTEST SUMMARY ==========")
    print(f"Trades:        {len(records)}")
    print(f"Win rate:      {win_rate}%")
    print(f"Total PnL:     ${total_pnl}")
    print(f"Total Return:  {total_return_pct}% of account")
    print(f"Avg Win:       ${avg_win}")
    print(f"Avg Loss:      ${avg_loss}")
    print("===========================================")


if __name__ == "__main__":
    run_backtest_v2(days_back=252)

