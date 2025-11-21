import datetime as dt
import math

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from config import ACCOUNT_EQUITY, RISK_PER_TRADE_PCT, UNIVERSE_FILE, SLACK_WEBHOOK_URL


############################################################
# Core risk / exit settings (same as V9/V10)
############################################################

STOP_MULT = 1.0    # stop = entry - STOP_MULT * ATR
TARGET_MULT = 2.0  # target = entry + TARGET_MULT * ATR


############################################################
# Basic helpers
############################################################

def load_universe():
    """Read tickers from the universe file (one per line)."""
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Universe file '{UNIVERSE_FILE}' not found.")
        return []


def fetch(ticker: str, start: dt.date, end: dt.date):
    """Download OHLCV data for a ticker via yfinance."""
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


def calc_atr(df: pd.DataFrame, period: int = 14):
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
# Market & sector regime filters
############################################################

def get_market_regime(as_of: dt.date, lookback_days: int = 40):
    """Simple SPY/QQQ trend + chop filter."""
    start = as_of - dt.timedelta(days=lookback_days)

    spy = fetch("SPY", start, as_of)
    qqq = fetch("QQQ", start, as_of)

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


def get_sector_risk_on(as_of: dt.date):
    """
    Very simple 'sector risk-on' check:
    looks at a few major sector ETFs and requires most of them to be trending up.
    """
    sector_etfs = ["XLY", "XLF", "XLK", "XLI", "XLV"]
    start = as_of - dt.timedelta(days=40)

    up_count = 0
    total = 0

    for etf in sector_etfs:
        df = fetch(etf, start, as_of)
        if df is None or len(df) < 20:
            continue

        close = df["Close"]
        last = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        day_ret = float((last - close.iloc[-2]) / close.iloc[-2])

        total += 1
        if last > ma20 and day_ret > 0:
            up_count += 1

    if total == 0:
        # if we couldn't get any data, don't hard-block trades
        return True

    # require at least half of the sectors to be in uptrend
    return up_count >= max(2, total // 2)


############################################################
# Relative strength vs SPY
############################################################

def get_relative_strength(stock_df: pd.DataFrame, spy_df: pd.DataFrame, as_of: dt.date, lookback: int = 5):
    """RS = stock return - SPY return over a short window."""
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
# V9-style scoring (no intraday ORB, for premarket watchlist)
############################################################

def score_ticker_v9_live(ticker: str, as_of: dt.date, df: pd.DataFrame, spy_df: pd.DataFrame):
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

    # Gap % — 0.6%–4.0%
    gap_pct = (last_close - prev_close) / prev_close
    if not (0.006 <= gap_pct <= 0.040):
        return None

    # Trend filter
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

    # RS vs SPY — allow up to ~1% underperformance
    rs = get_relative_strength(df, spy_df, as_of, lookback=5)
    if rs is None or rs <= -0.01:
        return None

    # VWAP-style confirmation
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    proxy_vwap = (last_high + last_low + last_close) / 3.0

    if last_close < proxy_vwap:
        return None

    score = 0.0

    # gap quality
    score += 40.0 * min(gap_pct * 100.0 / 2.0, 3.0)

    # relative volume
    if rel_vol >= 2.0:
        score += 25.0
    elif rel_vol >= 1.5:
        score += 15.0
    elif rel_vol >= 1.2:
        score += 5.0
    else:
        score -= 10.0

    # volatility sweet spot
    if 1.0 <= vol_score <= 4.0:
        score += 10.0
    elif vol_score > 6.0:
        score -= 5.0

    # trend strength
    trend_strength = (last_close - ma20) / ma20
    score += min(trend_strength * 200.0, 10.0)

    # RS bonus
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
    }


############################################################
# Position sizing (ATR-based, 2% risk per trade)
############################################################

def build_position_size(entry_price: float, atr: float):
    """Position size so that STOP_MULT * ATR ≈ 2% of account."""
    risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE_PCT
    stop_distance = STOP_MULT * atr

    if stop_distance <= 0:
        return None

    raw_qty = risk_dollars / stop_distance
    qty = max(1, math.floor(raw_qty))

    max_affordable_qty = math.floor(ACCOUNT_EQUITY / entry_price)
    if max_affordable_qty <= 0:
        return None

    qty = min(qty, max_affordable_qty)
    return qty


############################################################
# Slack helper
############################################################

def send_slack_message(text: str):
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
        if resp.status_code != 200:
            print(f"[SLACK] Error {resp.status_code}: {resp.text}")
        else:
            print("[SLACK] Message sent.")
    except Exception as e:
        print(f"[SLACK] Exception: {e}")


############################################################
# Trading calendar helper (SPY as source of "last trading day")
############################################################

def get_last_trading_day(today: dt.date):
    start = today - dt.timedelta(days=10)
    df = yf.download("SPY", start=start, end=today, progress=False, auto_adjust=False)
    if df.empty:
        # fallback: just use yesterday
        return today - dt.timedelta(days=1)
    dates = list(df.index.date)
    # pick the last date strictly before "today"
    candidates = [d for d in dates if d < today]
    if not candidates:
        return today - dt.timedelta(days=1)
    return candidates[-1]


############################################################
# Main routine
############################################################

def main():
    today = dt.date.today()
    signal_day = get_last_trading_day(today)

    print(f"[INFO] Running live V9 picker for signal day {signal_day}")

    # Skip weekends even if something weird happens
    if today.weekday() >= 5:
        print("[INFO] Weekend — no Slack picks will be sent.")
        return

    regime = get_market_regime(signal_day)
    sector_risk_on = get_sector_risk_on(signal_day)

    if not regime["allow_longs"] or not sector_risk_on:
        msg = (
            f"Good morning Ben – V9 model is **standing down** today.\n"
            f"Signal day: {signal_day}\n"
            f"Reason: Market/sector regime not supportive for longs.\n"
            f"- SPY/QQQ allow_longs: {regime['allow_longs']}\n"
            f"- Sector risk-on: {sector_risk_on}\n\n"
            "No trades recommended."
        )
        send_slack_message(msg)
        return

    universe = load_universe()
    if not universe:
        send_slack_message(
            "Good morning Ben – V9 model could not find any tickers in your universe file. "
            "No trades recommended."
        )
        return

    history_buffer_days = 90
    start_date = signal_day - dt.timedelta(days=history_buffer_days)

    print(f"[INFO] Downloading history {start_date} → {signal_day} for universe...")
    all_data = {}
    for t in universe:
        df = fetch(t, start_date, signal_day + dt.timedelta(days=1))
        if df is None:
            print(f"[WARN] No data for {t}")
            continue
        all_data[t] = df
        print(f"[DATA] {t}: {len(df)} rows")

    if not all_data:
        send_slack_message(
            "Good morning Ben – V9 model could not download data for any tickers. "
            "No trades recommended."
        )
        return

    # SPY history for RS
    spy_df = fetch("SPY", start_date, signal_day)
    if spy_df is None or spy_df.empty:
        send_slack_message(
            "Good morning Ben – V9 model could not fetch SPY for RS calculation. "
            "No trades recommended."
        )
        return

    signals = []

    for t, df in all_data.items():
        try:
            info = score_ticker_v9_live(t, signal_day, df, spy_df)
        except Exception as e:
            print(f"[WARN] Scoring failed for {t}: {e}")
            continue

        if info is not None:
            signals.append(info)

    if not signals:
        send_slack_message(
            f"Good morning Ben – V9 model found **no qualified trades** for {signal_day} "
            "after all filters.\nNo trades recommended."
        )
        return

    # Sort and select top 3 signals
    signals.sort(key=lambda x: x["score"], reverse=True)
    top_signals = signals[:3]

    # Build Slack message
    lines = []
    lines.append(
        "Good morning Ben – here are today’s **AI Stock Picks (V9/V10 model)** "
        f"based on signal day {signal_day}:"
    )
    lines.append("")
    lines.append(
        "_Backtest reference (paper): ~32.5% over 1 year, ~53% over 3 years at 2% risk per trade._"
    )
    lines.append("")
    lines.append(
        "_Execution note: Only consider entries if today’s price **breaks above yesterday’s high** "
        "(opening-range style confirmation)._"  # ORB guidance for you
    )
    lines.append("")

    for s in top_signals:
        price = s["price"]
        atr = s["atr"]
        gap_pct = s["gap_pct"] * 100.0
        rs_pct = s["rs"] * 100.0
        rel_vol = s["rel_vol"]

        qty = build_position_size(price, atr)
        stop_price = price - STOP_MULT * atr
        target_price = price + TARGET_MULT * atr

        line = (
            f"• *{s['ticker']}* @ ~${price:.2f}\n"
            f"  • Gap: {gap_pct:+.2f}%   RS vs SPY: {rs_pct:+.2f}%   RVOL: {rel_vol:.2f}x\n"
            f"  • ATR: ${atr:.2f}   Trend strength: {(s['trend_strength']*100):+.2f}%\n"
        )

        if qty is not None:
            line += (
                f"  • Suggested size: ~{qty} shares (risk ~{RISK_PER_TRADE_PCT*100:.1f}% of ${ACCOUNT_EQUITY:.0f})\n"
                f"  • Initial stop ≈ ${stop_price:.2f}  |  Initial target ≈ ${target_price:.2f}"
            )
        else:
            line += "  • Position sizing unavailable (check price vs account size)."

        lines.append(line)
        lines.append("")

    lines.append(
        "_This is a research / idea-generation tool, not investment advice. "
        "Always layer in your own judgment and risk management._"
    )

    msg = "\n".join(lines)
    print("[INFO] Final Slack message:\n", msg)
    send_slack_message(msg)


if __name__ == "__main__":
    main()

