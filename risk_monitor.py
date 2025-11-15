import datetime as dt
import json
import math
import os
import csv

import requests
import yfinance as yf
import pandas_market_calendars as mcal

from config import SLACK_WEBHOOK_URL, LOG_FILE

TRADE_STATE_FILE = "trade_state.json"


def debug(msg: str):
    print(f"[MONITOR DEBUG] {msg}")


def send_slack(text):
    """Send a simple text message to Slack via incoming webhook."""
    debug("send_slack called.")
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
    """Check if NYSE is open today."""
    if date is None:
        date = dt.datetime.now().date()

    debug(f"Checking if market open on {date}...")
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=date, end_date=date)
    is_open = not schedule.empty
    debug(f"Market open? {is_open}")
    return is_open


def load_state():
    """Load trade_state.json if it exists and is for today."""
    if not os.path.exists(TRADE_STATE_FILE):
        debug("No trade_state.json found.")
        return None

    try:
        with open(TRADE_STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception as e:
        debug(f"Error reading trade state: {e}")
        return None

    today_str = dt.datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today_str:
        debug("Trade state is for a previous date; ignoring.")
        return None

    if not state.get("active", True):
        debug("Trade state is inactive; nothing to monitor.")
        return None

    debug(f"Loaded active trade state: {state}")
    return state


def save_state(state):
    """Save updated trade state."""
    try:
        with open(TRADE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        debug("trade_state.json updated.")
    except Exception as e:
        debug(f"Error saving trade state: {e}")


def get_live_price(ticker):
    """Get the latest price for the ticker."""
    debug(f"Fetching live price for {ticker}...")
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            debug(f"Live price from 1m data: {price}")
            return price
    except Exception as e:
        debug(f"Error getting intraday price for {ticker}: {e}")

    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            debug(f"Fallback price from daily data: {price}")
            return price
    except Exception as e:
        debug(f"Error getting daily price for {ticker}: {e}")

    debug("Could not get price.")
    return None


def append_exit_to_log(state, exit_price, exit_reason):
    """Append a CLOSE record to trades.csv for this trade."""
    debug(f"Appending exit to log: reason={exit_reason}, price={exit_price}")
    file_exists = os.path.exists(LOG_FILE)

    row = {
        "date": state.get("date", ""),
        "ticker": state.get("ticker", ""),
        "side": state.get("side", ""),
        "label": state.get("label", ""),
        "entry": state.get("entry", ""),
        "stop": state.get("stop", ""),
        "target1": state.get("target1", ""),
        "target2": state.get("target2", ""),
        "atr": state.get("atr", ""),
        "atr_pct": state.get("atr_pct", ""),
        "risk_rating": state.get("risk_rating", ""),
        "qty": state.get("qty", ""),
        "status": "CLOSED",
        "exit_price": exit_price,
        "exit_reason": exit_reason,
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
        debug("Exit record appended to log.")
    except Exception as e:
        debug(f"Error appending exit to log: {e}")


def check_price_threat_and_profit(state):
    """Check stop, profit targets, and send alerts."""
    ticker = state["ticker"]
    label = state["label"]      # LONG / SHORT
    entry = state["entry"]
    stop = state["stop"]
    t1 = state["target1"]
    t2 = state["target2"]

    price = get_live_price(ticker)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if price is None:
        send_slack(f"⚠️ Monitor: Could not get live price for {ticker} at {now}.")
        return state

    state["last_price"] = price
    state["last_check"] = now

    stop_alert_sent = state.get("stop_alert_sent", False)
    t1_alert_sent = state.get("t1_alert_sent", False)
    t2_alert_sent = state.get("t2_alert_sent", False)

    # STOP HIT
    if label == "LONG" and price <= stop and not stop_alert_sent:
        send_slack(
            f"🚨 *Threat Alert — STOP Hit*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Side:* {label}\n"
            f"*Entry:* {entry}\n"
            f"*Stop:* {stop}\n"
            f"*Last Price:* {round(price, 2)}\n\n"
            f"Price has hit/broken the stop level. Consider closing the position to control risk."
        )
        state["stop_alert_sent"] = True
        state["active"] = False
        append_exit_to_log(state, round(price, 2), "STOP_HIT")
        return state

    if label == "SHORT" and price >= stop and not stop_alert_sent:
        send_slack(
            f"🚨 *Threat Alert — STOP Hit*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Side:* {label}\n"
            f"*Entry:* {entry}\n"
            f"*Stop:* {stop}\n"
            f"*Last Price:* {round(price, 2)}\n\n"
            f"Price has hit/broken the stop level. Consider closing the position to control risk."
        )
        state["stop_alert_sent"] = True
        state["active"] = False
        append_exit_to_log(state, round(price, 2), "STOP_HIT")
        return state

    # TARGET 1 HIT
    if label == "LONG" and price >= t1 and not t1_alert_sent:
        send_slack(
            f"✅ *Target 1 Reached*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Side:* {label}\n"
            f"*Entry:* {entry}\n"
            f"*Target 1:* {t1}\n"
            f"*Last Price:* {round(price, 2)}\n\n"
            f"Consider taking partial profits and moving stop to breakeven."
        )
        state["t1_alert_sent"] = True

    if label == "SHORT" and price <= t1 and not t1_alert_sent:
        send_slack(
            f"✅ *Target 1 Reached*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Side:* {label}\n"
            f"*Entry:* {entry}\n"
            f"*Target 1:* {t1}\n"
            f"*Last Price:* {round(price, 2)}\n\n"
            f"Consider taking partial profits and moving stop to breakeven."
        )
        state["t1_alert_sent"] = True

    # TARGET 2 HIT (full exit suggestion)
    if label == "LONG" and price >= t2 and not t2_alert_sent:
        send_slack(
            f"🎯 *Target 2 Reached — Full Profit Zone*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Side:* {label}\n"
            f"*Entry:* {entry}\n"
            f"*Target 2:* {t2}\n"
            f"*Last Price:* {round(price, 2)}\n\n"
            f"This is a logical area to close the position or lock in most of the gains."
        )
        state["t2_alert_sent"] = True
        state["active"] = False
        append_exit_to_log(state, round(price, 2), "TARGET2_HIT")
        return state

    if label == "SHORT" and price <= t2 and not t2_alert_sent:
        send_slack(
            f"🎯 *Target 2 Reached — Full Profit Zone*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Side:* {label}\n"
            f"*Entry:* {entry}\n"
            f"*Target 2:* {t2}\n"
            f"*Last Price:* {round(price, 2)}\n\n"
            f"This is a logical area to close the position or lock in most of the gains."
        )
        state["t2_alert_sent"] = True
        state["active"] = False
        append_exit_to_log(state, round(price, 2), "TARGET2_HIT")
        return state

    # Adverse move (> ~1% against entry)
    if label == "LONG":
        move_pct = (price - entry) / entry * 100.0
        if move_pct <= -1.0:
            send_slack(
                f"⚠️ *Play Under Pressure*\n\n"
                f"*Ticker:* {ticker}\n"
                f"*Side:* {label}\n"
                f"*Entry:* {entry}\n"
                f"*Last Price:* {round(price, 2)} ({round(move_pct, 2)}%)\n\n"
                f"Price has moved more than ~1% against the entry. Watch the trade closely."
            )
    else:  # SHORT
        if price >= entry * 1.01:
            move_pct = (price - entry) / entry * 100.0
            send_slack(
                f"⚠️ *Play Under Pressure*\n\n"
                f"*Ticker:* {ticker}\n"
                f"*Side:* {label}\n"
                f"*Entry:* {entry}\n"
                f"*Last Price:* {round(price, 2)} ({round(move_pct, 2)}%)\n\n"
                f"Price has moved more than ~1% against the entry. Watch the trade closely."
            )

    return state


def check_news_threat(state):
    """Check for fresh negative news and alert if needed."""
    ticker = state["ticker"]
    now = dt.datetime.now()
    now_ts = int(now.timestamp())

    debug(f"Checking news for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        news_items = getattr(t, "news", None)
    except Exception as e:
        debug(f"Error fetching news for {ticker}: {e}")
        return state

    if not news_items:
        debug("No news items returned.")
        return state

    bad_words = [
        "downgrade",
        "cut guidance",
        "misses",
        "sec",
        "probe",
        "investigation",
        "recall",
        "fraud",
        "lawsuit",
        "charge",
        "regulator",
        "warning",
    ]

    seen_ids = set(state.get("news_ids", []))
    new_seen_ids = set(seen_ids)

    for item in news_items:
        title = item.get("title", "").lower()
        ts = item.get("providerPublishTime") or item.get("published_at") or 0
        try:
            ts = int(ts)
        except Exception:
            continue

        # Only look at news from the last ~2 hours
        if now_ts - ts > 2 * 60 * 60:
            continue

        nid = item.get("uuid") or item.get("id") or (str(ts) + title[:20])
        if nid in seen_ids:
            continue

        if any(word in title for word in bad_words):
            pub_time = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            send_slack(
                f"📰 *News Alert — Potential Risk for {ticker}*\n\n"
                f"Headline: {item.get('title', 'N/A')}\n"
                f"Published: {pub_time}\n\n"
                f"Recent news may impact this trade. Consider reviewing the chart and risk."
            )
            new_seen_ids.add(nid)

    state["news_ids"] = list(new_seen_ids)
    return state


def maybe_end_of_day_exit(state):
    """
    Around 15:55–16:00 ET, suggest closing any remaining trade and log it.
    """
    now = dt.datetime.now()
    hour = now.hour
    minute = now.minute

    # Only act near the close
    if not (15 <= hour <= 16):
        return state

    if state.get("eod_alert_sent", False):
        return state

    price = state.get("last_price")
    if price is None:
        price = get_live_price(state["ticker"])

    if price is None:
        return state

    send_slack(
        f"🕒 *End-of-Day Exit Check*\n\n"
        f"*Ticker:* {state['ticker']}\n"
        f"*Side:* {state['label']}\n"
        f"*Entry:* {state['entry']}\n"
        f"*Last Price:* {round(price, 2)}\n\n"
        f"Market is near close. If you're still in this trade, this is a logical time to flatten the position."
    )
    state["eod_alert_sent"] = True
    state["active"] = False
    append_exit_to_log(state, round(price, 2), "TIME_EXIT")
    return state


def main():
    debug("=== Starting risk_monitor main() ===")
    now = dt.datetime.now()
    today = now.date()
    debug(f"Current datetime: {now}")

    # Skip if market closed (weekend/holiday)
    if not is_market_open_today(today):
        debug("Market closed; exiting.")
        return

    # Only run during regular hours via cron (we already constrained cron),
    # but we'll leave a sanity check:
    if not (9 <= now.hour <= 16):
        debug("Outside of 9–16h; exiting.")
        return

    state = load_state()
    if state is None:
        debug("No active trade to monitor; exiting.")
        return

    state = check_price_threat_and_profit(state)
    state = check_news_threat(state)
    state = maybe_end_of_day_exit(state)
    save_state(state)


if __name__ == "__main__":
    main()

