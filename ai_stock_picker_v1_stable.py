import datetime as dt
import math
import json
import os
import csv

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_market_calendars as mcal

from config import (
    SLACK_WEBHOOK_URL,
    ACCOUNT_EQUITY,
    RISK_PER_TRADE_PCT,
    UNIVERSE_FILE,
    LOG_FILE,
    RISK_MODE,
)

TRADE_STATE_FILE = "trade_state.json"


def debug(msg: str):
    """Simple debug printer."""
    print(f"[DEBUG] {msg}")


def load_universe():
    """Load ticker symbols from the universe file."""
    debug("Loading universe from file...")
    try:
        with open(UNIVERSE_FILE, "r") as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        debug(f"Universe file '{UNIVERSE_FILE}' not found.")
        return []
    debug(f"Universe loaded: {tickers}")
    return tickers


def get_history(ticker, days=40):
    """Download recent daily price history for a ticker."""
    debug(f"Downloading history for {ticker}...")
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days * 2)  # buffer for weekends/holidays
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,  # explicit to avoid FutureWarning changes
        )
    except Exception as e:
        debug(f"Error downloading data for {ticker}: {e}")
        return None

    if df.empty:
        debug(f"No data returned for {ticker}.")
        return None
    debug(f"Got {len(df)} rows for {ticker}.")
    return df.tail(days + 5)  # a few extra rows to ensure MA/ATR calc


def calc_atr(df, period=14):
    """Calculate Average True Range (ATR) using pandas."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def score_ticker(ticker):
    """Return a dict with signal info and score for a given ticker, or None."""
    df = get_history(ticker)
    if df is None:
        debug(f"Skipping {ticker}: no history.")
        return None

    df = df.dropna()
    if len(df) < 25:
        debug(f"Skipping {ticker}: not enough data after dropna.")
        return None

    close = df["Close"]

    last_close = close.iloc[-1].item()
    prev_close = close.iloc[-2].item()

    ma20_series = close.rolling(20).mean()
    ma20 = ma20_series.iloc[-1].item()

    # Trend filter
    trend = "long" if last_close > ma20 else "short"

    atr_series = calc_atr(df)
    atr = atr_series.iloc[-1].item()
    if math.isnan(atr) or atr <= 0:
        debug(f"Skipping {ticker}: ATR invalid ({atr}).")
        return None

    # using last vs prev close as "gap" proxy
    gap_pct = (last_close - prev_close) / prev_close

    # Base thresholds
    min_gap = 0.005  # 0.5%
    min_vol_score = 0.5

    # Adjust filters based on RISK_MODE
    if RISK_MODE == "conservative":
        min_gap = 0.0075
        min_vol_score = 0.7
    elif RISK_MODE == "aggressive":
        min_gap = 0.003
        min_vol_score = 0.3

    # Require minimum gap
    if abs(gap_pct) < min_gap:
        debug(f"Skipping {ticker}: gap {gap_pct:.4f} < {min_gap:.4f}.")
        return None

    # Volatility score (normalized ATR)
    vol_score = (atr / last_close) * 100.0
    if vol_score < min_vol_score:
        debug(f"Skipping {ticker}: vol_score {vol_score:.2f} < {min_vol_score:.2f}.")
        return None

    # Simple direction sanity: gap should agree with trend
    if gap_pct > 0 and trend != "long":
        debug(f"Skipping {ticker}: gap up but trend not long.")
        return None
    if gap_pct < 0 and trend != "short":
        debug(f"Skipping {ticker}: gap down but trend not short.")
        return None

    # Simple scoring formula
    score = 0.4 * abs(gap_pct) * 100.0 + 0.4 * vol_score + 0.2 * 1.0

    debug(
        f"Scored {ticker}: trend={trend}, price={last_close:.2f}, "
        f"ATR={atr:.2f}, gap_pct={gap_pct:.4f}, vol_score={vol_score:.2f}, score={score:.2f}"
    )

    return {
        "ticker": ticker,
        "trend": trend,       # "long" or "short"
        "price": float(last_close),
        "atr": float(atr),
        "gap_pct": float(gap_pct),
        "vol_score": float(vol_score),
        "score": float(score),
    }


def pick_best(universe):
    """Score all tickers and return the best one, or None."""
    debug("Picking best candidate from universe...")
    results = []
    for t in universe:
        info = score_ticker(t)
        if info is not None:
            results.append(info)

    if not results:
        debug("No candidates passed filters.")
        return None

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    debug(f"Best candidate: {best}")
    return best


def build_trade_plan(signal):
    """Given a signal dict, build a trade plan with entry/stop/targets/qty."""
    debug("Building trade plan...")
    price = signal["price"]
    atr = signal["atr"]
    risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE_PCT

    stop_distance = 1.5 * atr
    if stop_distance <= 0:
        debug(f"Invalid stop_distance: {stop_distance}")
        return None

    # Risk-based position sizing
    raw_qty = risk_dollars / stop_distance
    qty = max(1, math.floor(raw_qty))

    # Do not exceed what the account can actually buy/short
    max_affordable_qty = math.floor(ACCOUNT_EQUITY / price)
    if max_affordable_qty <= 0:
        debug("Max affordable qty <= 0.")
        return None
    qty = min(qty, max_affordable_qty)

    direction = "BUY" if signal["trend"] == "long" else "SELL"
    label = "LONG" if signal["trend"] == "long" else "SHORT"

    # Slight premium/discount on entry vs last price
    if label == "LONG":
        entry = round(price * 1.002, 2)
        stop = round(entry - stop_distance, 2)
        target1 = round(entry + 2 * stop_distance, 2)
        target2 = round(entry + 3 * stop_distance, 2)
    else:
        entry = round(price * 0.998, 2)
        stop = round(entry + stop_distance, 2)
        target1 = round(entry - 2 * stop_distance, 2)
        target2 = round(entry - 3 * stop_distance, 2)

    plan = {
        "direction": direction,  # BUY / SELL
        "label": label,          # LONG / SHORT
        "qty": qty,
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
    }
    debug(f"Trade plan: {plan}")
    return plan


def compute_risk_rating(signal):
    """Compute a simple risk rating based on normalized ATR."""
    price = signal["price"]
    atr = signal["atr"]
    atr_pct = (atr / price) * 100.0

    if atr_pct < 1.0:
        rating = "LOW"
    elif atr_pct < 2.0:
        rating = "MEDIUM"
    else:
        rating = "HIGH"

    return rating, round(atr_pct, 2)


def send_slack(text, context=""):
    """Send a simple text message to Slack via incoming webhook."""
    debug(f"send_slack called. Context='{context}'")
    if not SLACK_WEBHOOK_URL or SLACK_WEBHOOK_URL.startswith("PASTE_"):
        debug("SLACK_WEBHOOK_URL is not set correctly in config.py")
        debug("Message would have been:")
        debug(text)
        return

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        debug(f"Slack response: HTTP {resp.status_code} - {resp.text[:200]}")
        if resp.status_code != 200:
            print(f"Slack error: HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        debug(f"Error sending Slack message: {e}")


def is_market_open_today(date=None):
    """
    Check if the US stock market (NYSE) is open on the given date.
    Returns True if open, False if weekend/holiday/closed.
    """
    if date is None:
        date = dt.datetime.now().date()

    debug(f"Checking if market is open on {date}...")
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=date, end_date=date)
    is_open = not schedule.empty
    debug(f"Market open today? {is_open}")
    return is_open


def write_trade_state(trade_date, signal, plan, risk_rating, risk_atr_pct):
    """Save today's trade so the monitor script can watch it."""
    debug("Writing trade_state.json...")
    state = {
        "date": trade_date,
        "ticker": signal["ticker"],
        "trend": signal["trend"],           # "long"/"short"
        "side": plan["direction"],          # "BUY"/"SELL"
        "label": plan["label"],             # "LONG"/"SHORT"
        "entry": plan["entry"],
        "stop": plan["stop"],
        "target1": plan["target1"],
        "target2": plan["target2"],
        "atr": signal["atr"],
        "atr_pct": risk_atr_pct,
        "risk_rating": risk_rating,
        "qty": plan["qty"],
        "active": True,
        "stop_alert_sent": False,
        "t1_alert_sent": False,
        "t2_alert_sent": False,
        "eod_alert_sent": False,
        "last_price": None,
        "last_check": None,
        "news_ids": [],
    }
    try:
        with open(TRADE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        debug("trade_state.json written successfully.")
    except Exception as e:
        debug(f"Error writing trade state: {e}")


def append_trade_log(trade_date, signal, plan, risk_rating, risk_atr_pct):
    """Append this trade to trades.csv as an OPEN record."""
    debug("Appending trade to log...")
    file_exists = os.path.exists(LOG_FILE)
    row = {
        "date": trade_date,
        "ticker": signal["ticker"],
        "side": plan["direction"],
        "label": plan["label"],
        "entry": plan["entry"],
        "stop": plan["stop"],
        "target1": plan["target1"],
        "target2": plan["target2"],
        "atr": round(signal["atr"], 4),
        "atr_pct": risk_atr_pct,
        "risk_rating": risk_rating,
        "qty": plan["qty"],
        "status": "OPEN",
        "exit_price": "",
        "exit_reason": "",
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(LOG_FILE, "a", newline="") as csvfile:
            fieldnames = [
                "date",
                "ticker",
                "side",
                "label",
                "entry",
                "stop",
                "target1",
                "target2",
                "atr",
                "atr_pct",
                "risk_rating",
                "qty",
                "status",
                "exit_price",
                "exit_reason",
                "created_at",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        debug("Trade appended to log.")
    except Exception as e:
        debug(f"Error appending to trade log: {e}")


def main():
    debug("=== Starting ai_stock_picker main() ===")
    now = dt.datetime.now()
    today_date = now.date()
    today_str = now.strftime("%Y-%m-%d")
    debug(f"Current datetime: {now}")

    # Skip weekends & US market holidays
    if not is_market_open_today(today_date):
        msg = f"Good morning 🌅\n\nMarket is closed today ({today_str}). No new play."
        send_slack(msg, context="market_closed")
        debug("Exiting: market closed.")
        return

    universe = load_universe()
    if not universe:
        send_slack("AI Stock Picker: universe.txt is empty or missing. No tickers to scan.", context="empty_universe")
        debug("Exiting: universe empty.")
        return

    best = pick_best(universe)
    if best is None:
        send_slack(f"Good morning 🌅\n\nAI Stock Picker: No valid trade setups today ({today_str}).", context="no_setups")
        debug("Exiting: no valid setups.")
        return

    plan = build_trade_plan(best)
    if plan is None:
        send_slack(f"Good morning 🌅\n\nAI Stock Picker: Could not build trade plan for {today_str} - {best['ticker']}.", context="plan_failed")
        debug("Exiting: could not build trade plan.")
        return

    # Risk rating
    risk_rating, risk_atr_pct = compute_risk_rating(best)
    debug(f"Risk rating: {risk_rating}, ATR%: {risk_atr_pct}")

    # Save trade for monitor + log
    write_trade_state(today_str, best, plan, risk_rating, risk_atr_pct)
    append_trade_log(today_str, best, plan, risk_rating, risk_atr_pct)

    message = f"""
Good morning 🌅

📈 *AI Stock Picker — Daily Play for {today_str}*

*Ticker:* {best['ticker']}
*Direction:* {plan['label']}
*Entry:* {plan['entry']}
*Stop:* {plan['stop']}
*Target 1:* {plan['target1']}
*Target 2:* {plan['target2']}
*Qty:* {plan['qty']}

*Risk profile:*
- Risk mode: {RISK_MODE.upper()}
- Volatility (ATR%): {risk_atr_pct}%
- Risk rating: *{risk_rating}*

*Moomoo Order Settings:*
- Side: {plan['direction']}
- Order Type: LIMIT
- Quantity: {plan['qty']}
- Limit Price: {plan['entry']}
- Time-in-force: DAY
- Session: Regular Hours Only

*Exit plan (manual guidance):*
- *Threat exit:* If price clearly breaks the stop level and holds there, close the position.
- *Profit exits:* 
  - Take partial profits around Target 1.
  - Take remaining size around Target 2 or if momentum stalls.
- *Time exit:* Close any remaining position by ~3:55 PM ET at market price.
- *Something feels wrong:* If news drops, spread blows out, liquidity vanishes, or the trade feels broken, it's okay to flatten early.

_This is not financial advice. Trade at your own risk._
"""
    send_slack(message, context="daily_play")
    debug("Daily play sent. Exiting main().")


if __name__ == "__main__":
    main()

