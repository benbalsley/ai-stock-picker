import datetime as dt
import math
import csv

import numpy as np
import pandas as pd
import yfinance as yf

from config import ACCOUNT_EQUITY, RISK_PER_TRADE_PCT, UNIVERSE_FILE


############################################################
# Universe + Data Helpers
############################################################

def load_universe():
    """Load tickers from universe file."""
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Universe file '{UNIVERSE_FILE}' not found.")
        return []


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
# Market Regime (SPY + QQQ)
############################################################

def get_market_regime(end_date: dt.date, lookback_days: int = 40):
    """
    Use SPY & QQQ to determine if the market is supportive for LONGS.
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
# Relative Strength vs SPY
############################################################

def get_relative_strength(stock_df: pd.DataFrame, spy_df: pd.DataFrame, as_of: dt.date, lookback: int = 5):
    """
    Compute 5-day relative strength: stock return - SPY return.
    Positive = stock stronger than SPY.
    """
    stock_up_to = stock_df[stock_df.index.date <= as_of].copy()
    spy_up_to = spy_df[spy_df.index.date <= as_of].copy()

    if len(stock_up_to) < lookback + 2 or len(spy_up_to) < lookback + 2:
        return None

    stock_close = stock_up_to["Close"]
    spy_close = spy_up_to["Close"]

    stock_end = float(stock_close.iloc[-1])
    stock_start = float(stock_close.iloc[-lookback - 1])

    spy_end = float(spy_close.iloc[-1])
    spy_start = float(spy_close.iloc[-lookback - 1])

    stock_ret = (stock_end - stock_start) / stock_start
    spy_ret = (spy_end - spy_start) / spy_start

    return stock_ret - spy_ret


############################################################
# Scoring: V5 logic (VWAP + RS + filters)
############################################################

def score_ticker_v6(ticker: str, as_of: dt.date, df: pd.DataFrame, spy_df: pd.DataFrame):
    """
    V6 scoring:
      - Long-only
      - Gap sweet spot (0.8%–3.5%)
      - Price above MA20
      - Liquidity (avg vol > 1M)
      - Relative volume
      - Volatility sanity
      - Relative strength vs SPY > 0
      - Close above proxy VWAP (typical price) on signal day
    """
    df_up_to = df[df.index.date <= as_of].copy()
    df_up_to = df_up_to.dropna()

    if len(df_up_to) < 30:
        return None

    close = df_up_to["Close"]
    vol = df_up_to["Volume"]
    high = df_up_to["High"]
    low = df_up_to["Low"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    ma20 = float(close.rolling(20).mean().iloc[-1])

    atr_series = calc_atr(df_up_to)
    atr = float(atr_series.iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return None

    # Gap %
    gap_pct = (last_close - prev_close) / prev_close
    if not (0.008 <= gap_pct <= 0.035):
        return None

    # Trend
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

    # Relative strength vs SPY
    rs = get_relative_strength(df, spy_df, as_of, lookback=5)
    if rs is None or rs <= 0:
        return None

    # VWAP-style confirmation: close above proxy VWAP on signal day
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    proxy_vwap = (last_high + last_low + last_close) / 3.0

    if last_close < proxy_vwap:
        return None

    # Build score
    score = 0.0

    # 1) Gap quality
    score += 40.0 * min(gap_pct * 100.0 / 2.0, 3.0)

    # 2) Relative volume
    if rel_vol >= 2.0:
        score += 25.0
    elif rel_vol >= 1.5:
        score += 15.0
    elif rel_vol >= 1.2:
        score += 5.0
    else:
        score -= 10.0

    # 3) Volatility sweet spot
    if 1.0 <= vol_score <= 4.0:
        score += 10.0
    elif vol_score > 6.0:
        score -= 5.0

    # 4) Trend strength
    trend_strength = (last_close - ma20) / ma20
    score += min(trend_strength * 200.0, 10.0)

    # 5) Relative strength bonus
    score += min(rs * 200.0, 10.0)

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
        "rs": rs,
        "score": score,
        "proxy_vwap": proxy_vwap,
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
    return list(df.index.date), df


def build_position_size(entry_price, atr, stop_mult):
    """Risk per trade based on stop_mult * ATR stop distance."""
    risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE_PCT
    stop_distance = stop_mult * atr

    if stop_distance <= 0:
        return None

    raw_qty = risk_dollars / stop_distance
    qty = max(1, math.floor(raw_qty))

    max_affordable_qty = math.floor(ACCOUNT_EQUITY / entry_price)
    if max_affordable_qty <= 0:
        return None

    qty = min(qty, max_affordable_qty)
    return qty


def simulate_trade_for_day(signal_date, signal, df, trading_days, stop_mult, target_mult):
    """
    Entry: next day open
    Exit logic (long-only):
      - Stop = entry - stop_mult * ATR
      - Target = entry + target_mult * ATR
      - If low <= stop AND high >= target → assume stop hit first (conservative)
      - Else if high >= target → exit at target
      - Else if low <= stop → exit at stop
      - Else → exit at close
    """
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
    high_px = float(df_trade["High"].iloc[0])
    low_px = float(df_trade["Low"].iloc[0])
    close_px = float(df_trade["Close"].iloc[0])

    atr = signal["atr"]
    entry_price = open_px
    stop_price = entry_price - stop_mult * atr
    target_price = entry_price + target_mult * atr

    qty = build_position_size(entry_price, atr, stop_mult)
    if qty is None:
        return None

    if low_px <= stop_price and high_px >= target_price:
        exit_price = stop_price
        exit_reason = "stop_and_target_same_day_stop_first"
    elif high_px >= target_price:
        exit_price = target_price
        exit_reason = "target_hit"
    elif low_px <= stop_price:
        exit_price = stop_price
        exit_reason = "stop_hit"
    else:
        exit_price = close_px
        exit_reason = "close_exit"

    pnl_per_share = exit_price - entry_price
    pnl_dollars = pnl_per_share * qty
    pnl_pct = pnl_dollars / ACCOUNT_EQUITY * 100.0

    return {
        "signal_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "ticker": signal["ticker"],
        "trend": "long",
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "qty": qty,
        "pnl_dollars": round(pnl_dollars, 2),
        "pnl_pct": round(pnl_pct, 4),
        "gap_pct": round(signal["gap_pct"] * 100.0, 2),
        "atr": round(signal["atr"], 4),
        "rel_vol": round(signal["rel_vol"], 4),
        "vol_score": round(signal["vol_score"], 4),
        "rs": round(signal["rs"], 4),
        "score": round(signal["score"], 4),
        "exit_reason": exit_reason,
    }


############################################################
# Core backtest for a single (stop_mult, target_mult)
############################################################

def run_backtest_once(days_back, stop_mult, target_mult, top_n=3):
    today = dt.date.today()

    history_buffer_days = 90
    start_date = today - dt.timedelta(days=days_back + history_buffer_days)
    end_date = today

    print(f"\n[CONFIG] STOP_MULT={stop_mult}, TARGET_MULT={target_mult}, top_n={top_n}")
    print(f"[INFO] V6 window: last {days_back} trading days ({start_date} → {end_date})")

    trading_days, spy_df = get_spy_trading_days(start_date, end_date)

    if len(trading_days) <= days_back + 1:
        signal_days = trading_days[:-1]
    else:
        signal_days = trading_days[-(days_back + 1):-1]

    universe = load_universe()
    if not universe:
        print("[ERROR] Universe is empty. Aborting V6 backtest.")
        return None

    all_data = download_history_for_universe(universe, start_date, end_date)

    records = []

    for d in signal_days:
        regime = get_market_regime(d)
        if not regime["allow_longs"]:
            continue

        day_signals = []
        for t, df in all_data.items():
            try:
                info = score_ticker_v6(t, d, df, spy_df)
            except Exception as e:
                print(f"[WARN] Score failed for {t} on {d}: {e}")
                continue
            if info is None:
                continue
            day_signals.append(info)

        if not day_signals:
            continue

        day_signals.sort(key=lambda x: x["score"], reverse=True)
        selected = day_signals[:top_n]

        for signal in selected:
            df_t = all_data[signal["ticker"]]
            rec = simulate_trade_for_day(d, signal, df_t, trading_days, stop_mult, target_mult)
            if rec is None:
                continue
            records.append(rec)

    if not records:
        print("[RESULT] No trades for this configuration.")
        return None

    pnl_series = [r["pnl_dollars"] for r in records]
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p <= 0]

    total_pnl = sum(pnl_series)
    total_return_pct = sum(r["pnl_pct"] for r in records)
    win_rate = (len(wins) / len(records) * 100.0) if records else 0.0

    summary = {
        "stop_mult": stop_mult,
        "target_mult": target_mult,
        "trades": len(records),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }

    print("  Trades:       ", summary["trades"])
    print("  Win rate:     ", f"{summary['win_rate']:.2f}%")
    print("  Total PnL:    ", f"${summary['total_pnl']:.2f}")
    print("  Total Return: ", f"{summary['total_return_pct']:.2f}% of account")
    print("  Avg Win:      ", f"${summary['avg_win']:.2f}" if wins else "N/A")
    print("  Avg Loss:     ", f"${summary['avg_loss']:.2f}" if losses else "N/A")

    return summary


############################################################
# Grid runner: try many exit combos and print best
############################################################

def run_backtest_v6():
    days_back = 252
    top_n = 3

    stop_values = [0.8, 1.0, 1.2]
    target_values = [1.5, 2.0, 2.5]

    results = []

    for s in stop_values:
        for t in target_values:
            summary = run_backtest_once(days_back, s, t, top_n=top_n)
            if summary is not None:
                results.append(summary)

    if not results:
        print("\n[RESULT] No configurations produced trades.")
        return

    # Find best by total_return_pct
    best = max(results, key=lambda x: x["total_return_pct"])

    print("\n========== V6 EXIT GRID SUMMARY ==========")
    for res in results:
        print(
            f"STOP={res['stop_mult']:.2f}, TARGET={res['target_mult']:.2f} "
            f"-> Trades={res['trades']}, Win={res['win_rate']:.2f}%, "
            f"Return={res['total_return_pct']:.2f}%"
        )
    print("===========================================")
    print(
        f"BEST CONFIG -> STOP_MULT={best['stop_mult']:.2f}, "
        f"TARGET_MULT={best['target_mult']:.2f}, "
        f"Return={best['total_return_pct']:.2f}% on account"
    )


if __name__ == "__main__":
    run_backtest_v6()

