import datetime as dt
import math
import csv

import numpy as np
import pandas as pd
import yfinance as yf

from config import ACCOUNT_EQUITY, RISK_PER_TRADE_PCT, UNIVERSE_FILE


############################################################
# Helpers: data + indicators
############################################################

def load_universe():
    """Load tickers from universe file."""
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Universe file '{UNIVERSE_FILE}' not found.")
        return []


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
    Use SPY daily data as the trading calendar (simpler & robust).
    """
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Could not get SPY history to build trading calendar.")
    return list(df.index.date)


def calc_atr(df, period=14):
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


############################################################
# Scoring logic (mirror live picker V1)
############################################################

def score_on_date(ticker, df, date):
    """
    Score a ticker as of 'date' using the same logic as the live picker.
    df: full daily DataFrame for ticker
    date: datetime.date (signal date, we use close of that day)
    """
    # Use data up to and including 'date'
    df_up_to = df[df.index.date <= date].copy()
    df_up_to = df_up_to.dropna()

    if len(df_up_to) < 25:
        return None

    close = df_up_to["Close"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    ma20_series = close.rolling(20).mean()
    ma20 = float(ma20_series.iloc[-1])

    # Trend filter
    trend = "long" if last_close > ma20 else "short"

    atr_series = calc_atr(df_up_to)
    atr = float(atr_series.iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return None

    # Gap between last and previous close
    gap_pct = (last_close - prev_close) / prev_close

    # Minimum gap requirement (~0.5%)
    if abs(gap_pct) < 0.005:
        return None

    # Normalized ATR as volatility measure
    vol_score = (atr / last_close) * 100.0
    if vol_score < 0.5:
        return None

    # Direction sanity: gap should agree with trend
    if gap_pct > 0 and trend != "long":
        return None
    if gap_pct < 0 and trend != "short":
        return None

    # Simple scoring formula (same spirit as live code)
    score = 0.4 * abs(gap_pct) * 100.0 + 0.4 * vol_score + 0.2 * 1.0

    return {
        "ticker": ticker,
        "trend": trend,
        "price": float(last_close),
        "atr": float(atr),
        "gap_pct": float(gap_pct),
        "vol_score": float(vol_score),
        "score": float(score),
    }


############################################################
# Trade simulation (simple: enter next open, exit same close)
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

    # Row for trade_date from df
    df_trade = df[df.index.date == trade_date]
    if df_trade.empty:
        return None

    open_px = float(df_trade["Open"].iloc[0])
    close_px = float(df_trade["Close"].iloc[0])

    qty = build_position_size(signal)
    if qty is None:
        return None

    direction = 1 if signal["trend"] == "long" else -1

    pnl_per_share = (close_px - open_px) * direction
    pnl_dollars = pnl_per_share * qty
    pnl_pct = pnl_dollars / ACCOUNT_EQUITY * 100.0

    record = {
        "signal_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "ticker": signal["ticker"],
        "trend": signal["trend"],
        "entry_price": round(open_px, 4),
        "exit_price": round(close_px, 4),
        "qty": qty,
        "pnl_dollars": round(pnl_dollars, 2),
        "pnl_pct": round(pnl_pct, 4),
        "gap_pct": round(signal["gap_pct"] * 100.0, 2),
        "atr": round(signal["atr"], 4),
        "vol_score": round(signal["vol_score"], 4),
        "score": round(signal["score"], 4),
    }
    return record


############################################################
# Backtest driver
############################################################

def run_backtest(days_back=252, output_file="backtest_results.csv"):
    """
    Backtest roughly 1 year of signals:
    - One signal per trading day (best candidate)
    - Enter next day open
    - Exit same day close
    """
    today = dt.date.today()

    # Use a wider start for history (ATR, MA) + backtest window
    history_buffer_days = 80
    start_date = today - dt.timedelta(days=days_back + history_buffer_days)
    end_date = today

    print(f"[INFO] Backtest window (signals): last {days_back} trading days")
    print(f"[INFO] History pulled from {start_date} to {end_date}")

    # Trading calendar from SPY
    trading_days = get_spy_trading_days(start_date, end_date)
    if len(trading_days) < days_back + 10:
        print("[WARN] Fewer trading days than expected; check SPY data.")

    # Use the last `days_back` days as the signal dates
    if len(trading_days) <= days_back + 1:
        signal_days = trading_days[:-1]
    else:
        signal_days = trading_days[-(days_back + 1):-1]  # leave last day for next trade

    universe = load_universe()
    if not universe:
        print("[ERROR] Universe is empty. Aborting backtest.")
        return

    print(f"[INFO] Universe size: {len(universe)} tickers")

    # Download all history once
    all_data = download_history_for_universe(universe, start_date, end_date)

    # Prepare CSV
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
        "vol_score",
        "score",
    ]
    records = []

    for d in signal_days:
        print(f"\n[DAY] Processing signal for {d}...")

        best_signal = None

        for t, df in all_data.items():
            score = score_on_date(t, df, d)
            if score is None:
                continue
            if (best_signal is None) or (score["score"] > best_signal["score"]):
                best_signal = score

        if best_signal is None:
            print(f"[DAY] {d}: no valid signal")
            continue

        df_t = all_data[best_signal["ticker"]]
        rec = simulate_trade_for_day(d, best_signal, df_t, trading_days)
        if rec is None:
            print(f"[DAY] {d}: could not simulate trade for {best_signal['ticker']}")
            continue

        print(
            f"[TRADE] {rec['trade_date']} {rec['ticker']} {rec['trend'].upper()} "
            f"entry={rec['entry_price']} exit={rec['exit_price']} "
            f"PnL=${rec['pnl_dollars']} ({rec['pnl_pct']}%)"
        )

        records.append(rec)

    # Write CSV
    if records:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"\n[RESULT] Backtest written to {output_file}")
    else:
        print("\n[RESULT] No trades generated. Nothing written.")

    # Simple stats
    if not records:
        return

    pnl_series = [r["pnl_dollars"] for r in records]
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p <= 0]

    total_pnl = round(sum(pnl_series), 2)
    total_return_pct = round(sum(r["pnl_pct"] for r in records), 2)
    win_rate = round(len(wins) / len(records) * 100.0, 2) if records else 0.0
    avg_win = round(np.mean(wins), 2) if wins else 0.0
    avg_loss = round(np.mean(losses), 2) if losses else 0.0

    print("\n========== BACKTEST SUMMARY ==========")
    print(f"Trades:        {len(records)}")
    print(f"Win rate:      {win_rate}%")
    print(f"Total PnL:     ${total_pnl}")
    print(f"Total Return:  {total_return_pct}% of account")
    print(f"Avg Win:       ${avg_win}")
    print(f"Avg Loss:      ${avg_loss}")
    print("=======================================")


if __name__ == "__main__":
    # ~252 trading days ≈ 1 year
    run_backtest(days_back=252)

