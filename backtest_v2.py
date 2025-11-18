import datetime as dt
import math
import csv

import numpy as np
import pandas as pd
import yfinance as yf

from config import ACCOUNT_EQUITY, RISK_PER_TRADE_PCT, UNIVERSE_FILE


############################################################
# Load Universe
############################################################

def load_universe():
    """Load tickers from universe file."""
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Universe file '{UNIVERSE_FILE}' not found.")
        return []


############################################################
# Fetch + Indicators
############################################################

def fetch(ticker, start, end):
    """Download daily bars for a ticker."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return None
    return df.dropna()


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
# FIXED + CLEAN — MARKET REGIME LOGIC (SCALAR ONLY)
############################################################

def get_market_regime(end_date: dt.date, lookback_days: int = 40):
    """
    Use SPY & QQQ to determine if the market is supportive for LONGS.
    Clean scalar-only version (no ambiguous Series comparisons).
    """
    start = end_date - dt.timedelta(days=lookback_days)

    spy = fetch("SPY", start, end_date)
    qqq = fetch("QQQ", start, end_date)

    if spy is None or qqq is None or len(spy) < 20 or len(qqq) < 20:
        print("[WARN] Could not get full SPY/QQQ data, defaulting to neutral regime.")
        return {"allow_longs": True}

    spy_close = spy["Close"]
    qqq_close = qqq["Close"]

    spy_last = float(spy_close.iloc[-1])
    qqq_last = float(qqq_close.iloc[-1])

    spy_ma20 = float(spy_close.rolling(20).mean().iloc[-1])
    qqq_ma20 = float(qqq_close.rolling(20).mean().iloc[-1])

    spy_ret = float((spy_last - spy_close.iloc[-2]) / spy_close.iloc[-2])
    qqq_ret = float((qqq_last - qqq_close.iloc[-2]) / qqq_close.iloc[-2])

    spy_up = spy_last > spy_ma20
    qqq_up = qqq_last > qqq_ma20

    spy_range_pct = float((spy["High"].iloc[-1] - spy["Low"].iloc[-1]) / spy_last)
    qqq_range_pct = float((qqq["High"].iloc[-1] - qqq["Low"].iloc[-1]) / qqq_last)

    chop = (spy_range_pct < 0.002) and (qqq_range_pct < 0.002)

    allow_longs = (spy_up and qqq_up and spy_ret > 0 and qqq_ret > 0 and not chop)

    return {
        "allow_longs": allow_longs,
        "spy_up_trend": spy_up,
        "qqq_up_trend": qqq_up,
        "spy_ret": spy_ret,
        "qqq_ret": qqq_ret,
        "chop": chop,
    }


############################################################
# V2 Institutional Scoring
############################################################

def score_ticker_v2(ticker: str, as_of: dt.date, df: pd.DataFrame):
    df_up_to = df[df.index.date <= as_of].copy()
    df_up_to = df_up_to.dropna()

    if len(df_up_to) < 25:
        return None

    close = df_up_to["Close"]
    vol = df_up_to["Volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    ma20 = float(close.rolling(20).mean().iloc[-1])

    atr_series = calc_atr(df_up_to)
    atr = float(atr_series.iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return None

    gap_pct = (last_close - prev_close) / prev_close

    # Institutional sweet spot: 0.8%–3.5% gap
    if not (0.008 <= gap_pct <= 0.035):
        return None

    # Trend requirement
    if last_close <= ma20:
        return None

    avg_vol20 = float(vol.rolling(20).mean().iloc[-1])
    today_vol = float(vol.iloc[-1])

    if avg_vol20 < 1_000_000:
        return None

    rel_vol = today_vol / avg_vol20 if avg_vol20 > 0 else 0.0

    vol_score = (atr / last_close) * 100.0
    if vol_score < 0.5:
        return None

    score = 0.0

    score += 40.0 * min(gap_pct * 100.0 / 2.0, 3.0)

    if rel_vol >= 2.0:
        score += 25.0
    elif rel_vol >= 1.5:
        score += 15.0
    elif rel_vol >= 1.2:
        score += 5.0
    else:
        score -= 10.0

    if 1.0 <= vol_score <= 4.0:
        score += 10.0
    elif vol_score > 6.0:
        score -= 5.0

    trend_strength = (last_close - ma20) / ma20
    score += min(trend_strength * 200.0, 10.0)

    return {
        "ticker": ticker,
        "as_of": as_of,
        "price": last_close,
        "gap_pct": gap_pct,
        "ma20": ma20,
        "atr": atr,
        "avg_vol20": avg_vol20,
        "today_vol": today_vol,
        "rel_vol": rel_vol,
        "vol_score": vol_score,
        "trend_strength": trend_strength,
        "score": score,
    }


############################################################
# Backtest Helpers
############################################################

def download_history_for_universe(tickers, start, end):
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
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Could not get SPY history to build trading calendar.")
    return list(df.index.date)


def build_position_size(signal):
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
    try:
        idx = trading_days.index(signal_date)
    except ValueError:
        return None

    if idx + 1 >= len(trading_days):
        return None

    trade_date = trading_days[idx + 1]

    df_trade = df[df.index.date == trade_date]
    if df_trade.empty:
        return None

    open_px = float(df_trade["Open"].iloc[0])
    close_px = float(df_trade["Close"].iloc[0])

    qty = build_position_size(signal)
    if qty is None:
        return None

    pnl_per_share = (close_px - open_px)
    pnl_dollars = pnl_per_share * qty
    pnl_pct = pnl_dollars / ACCOUNT_EQUITY * 100.0

    return {
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


############################################################
# Main Backtest Loop
############################################################

def run_backtest_v2(days_back=252, output_file="backtest_results_v2.csv"):
    today = dt.date.today()

    history_buffer_days = 90
    start_date = today - dt.timedelta(days=days_back + history_buffer_days)
    end_date = today

    print(f"[INFO] V2 Backtest window: last {days_back} trading days")
    print(f"[INFO] Pulling history {start_date} → {end_date}")

    trading_days = get_spy_trading_days(start_date, end_date)

    if len(trading_days) <= days_back + 1:
        signal_days = trading_days[:-1]
    else:
        signal_days = trading_days[-(days_back + 1):-1]

    universe = load_universe()
    if not universe:
        print("[ERROR] Universe is empty. Aborting backtest.")
        return

    print(f"[INFO] Universe: {len(universe)} tickers")

    all_data = download_history_for_universe(universe, start_date, end_date)

    fieldnames = [
        "signal_date", "trade_date", "ticker", "trend",
        "entry_price", "exit_price", "qty",
        "pnl_dollars", "pnl_pct",
        "gap_pct", "atr", "rel_vol", "vol_score", "score",
    ]

    records = []

    for d in signal_days:
        print(f"\n[DAY] {d} — computing V2 signal...")

        # Market regime
        regime = get_market_regime(d)
        if not regime["allow_longs"]:
            print("[DAY] Market regime blocked longs today.")
            continue

        best_signal = None

        for t, df in all_data.items():
            try:
                info = score_ticker_v2(t, d, df)
            except Exception as e:
                print(f"[WARN] Score failed for {t} on {d}: {e}")
                continue

            if info is None:
                continue

            if (best_signal is None) or (info["score"] > best_signal["score"]):
                best_signal = info

        if best_signal is None:
            print(f"[DAY] {d}: No V2 signals passed filters.")
            continue

        df_t = all_data[best_signal["ticker"]]
        rec = simulate_trade_for_day(d, best_signal, df_t, trading_days)

        if rec is None:
            print(f"[DAY] {d}: trade simulation failed.")
            continue

        print(
            f"[TRADE] {rec['trade_date']} {rec['ticker']} "
            f"entry={rec['entry_price']} exit={rec['exit_price']} "
            f"PnL=${rec['pnl_dollars']} ({rec['pnl_pct']}%)"
        )

        records.append(rec)

    if records:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"\n[RESULT] Backtest V2 written to {output_file}")
    else:
        print("\n[RESULT] No V2 trades generated.")
        return

    # Summary
    pnl_series = [r["pnl_dollars"] for r in records]
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p <= 0]

    print("\n========== V2 BACKTEST SUMMARY ==========")
    print(f"Trades:        {len(records)}")
    print(f"Win rate:      {len(wins) / len(records) * 100:.2f}%")
    print(f"Total PnL:     ${sum(pnl_series):.2f}")
    print(f"Total Return:  {sum(r['pnl_pct'] for r in records):.2f}% of account")
    print(f"Avg Win:       ${np.mean(wins):.2f}" if wins else "Avg Win:       N/A")
    print(f"Avg Loss:      ${np.mean(losses):.2f}" if losses else "Avg Loss:      N/A")
    print("===========================================")


if __name__ == "__main__":
    run_backtest_v2(days_back=252)

