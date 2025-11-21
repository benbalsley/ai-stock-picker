import os
import csv
import re
import datetime as dt

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import yfinance as yf


# Env vars provided by GitHub Actions / your local shell
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")

TRADES_FILE = "executed_trades.csv"


def get_client():
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set.")
    return WebClient(token=SLACK_BOT_TOKEN)


def load_trades():
    if not os.path.exists(TRADES_FILE):
        return []

    trades = []
    with open(TRADES_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades


def save_trades(trades):
    fieldnames = [
        "Ticker",
        "Direction",
        "EntryPrice",
        "EntryTimeUTC",
        "ExitPrice",
        "ExitTimeUTC",
        "PctReturn",
        "EntryMessageTS",
        "ExitMessageTS",
    ]
    with open(TRADES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(t)


def parse_tickers_from_text(text):
    """
    For lines like:
    • *AAPL* @ ~$190.23
    or
    • AAPL (long) @ ...
    we try to grab the ticker.
    """
    tickers = set()

    # Look for patterns like *AAPL*
    for match in re.findall(r"\*([A-Z\.]{1,6})\*", text):
        tickers.add(match.strip().upper())

    # Fallback: lines starting with bullet and plain ticker
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("• "):
            # Try to parse first word after bullet
            parts = line[2:].split()
            if parts:
                candidate = parts[0].strip("*").strip().upper()
                if candidate.isalpha() and 1 <= len(candidate) <= 6:
                    tickers.add(candidate)

    return sorted(tickers)


def get_current_price(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=False)
        if df.empty:
            # fallback to daily
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if df.empty:
                return None
        close = df["Close"]
        return float(close.iloc[-1])
    except Exception as e:
        print(f"[PRICE] Error for {ticker}: {e}")
        return None


def has_check_reaction(message):
    """
    Returns True if the message has a ✅ / white_check_mark reaction.
    """
    reactions = message.get("reactions", [])
    for r in reactions:
        name = r.get("name", "")
        if name in ("white_check_mark", "heavy_check_mark"):
            return True
    return False


def is_entry_message(message):
    text = message.get("text", "")
    return "AI Stock Picks (V9/V10 model)" in text or "AI Stock Picks" in text


def is_exit_message(message):
    text = message.get("text", "")
    return "Risk Monitor V9" in text


def sync_entries_from_message(message, trades):
    """
    If this entry message has a ✅ and hasn't been used yet, log entries
    for each ticker at current market price.
    """
    ts = message["ts"]
    text = message.get("text", "")

    # Already used as entry?
    for t in trades:
        if t.get("EntryMessageTS") == ts:
            # Already processed this message
            return trades

    if not has_check_reaction(message):
        return trades

    tickers = parse_tickers_from_text(text)
    if not tickers:
        print(f"[ENTRY] No tickers found in message {ts}")
        return trades

    now = dt.datetime.utcnow().isoformat()
    print(f"[ENTRY] Confirmed entry via ✅ on message {ts}, tickers={tickers}")

    for ticker in tickers:
        price = get_current_price(ticker)
        if price is None:
            print(f"[ENTRY] Skipping {ticker}: no current price")
            continue

        trade = {
            "Ticker": ticker,
            "Direction": "long",   # V1: assume long entries; can refine later
            "EntryPrice": f"{price:.4f}",
            "EntryTimeUTC": now,
            "ExitPrice": "",
            "ExitTimeUTC": "",
            "PctReturn": "",
            "EntryMessageTS": ts,
            "ExitMessageTS": "",
        }
        trades.append(trade)
        print(f"[ENTRY] Logged {ticker} entry at {price:.4f} (UTC {now})")

    return trades


def sync_exits_from_message(message, trades):
    """
    If this exit (risk monitor) message has a ✅, close any open trades
    for tickers mentioned in the message, using current market price.
    """
    ts = message["ts"]
    text = message.get("text", "")

    if not has_check_reaction(message):
        return trades

    tickers = parse_tickers_from_text(text)
    if not tickers:
        print(f"[EXIT] No tickers found in message {ts}")
        return trades

    now = dt.datetime.utcnow().isoformat()
    print(f"[EXIT] Confirmed exit via ✅ on message {ts}, tickers={tickers}")

    # For each ticker, close most recent open trade
    for ticker in tickers:
        current_price = get_current_price(ticker)
        if current_price is None:
            print(f"[EXIT] Skipping {ticker}: no current price")
            continue

        # Find most recent open trade for this ticker
        open_trades = [
            (idx, t) for idx, t in enumerate(trades)
            if t["Ticker"] == ticker and not t.get("ExitPrice")
        ]
        if not open_trades:
            print(f"[EXIT] No open trade found for {ticker}")
            continue

        # Use the last one as "most recent"
        idx, t = open_trades[-1]

        entry_price = float(t["EntryPrice"])
        direction = t.get("Direction", "long")

        if direction == "long":
            pct_return = (current_price - entry_price) / entry_price * 100.0
        else:
            pct_return = (entry_price - current_price) / entry_price * 100.0

        trades[idx]["ExitPrice"] = f"{current_price:.4f}"
        trades[idx]["ExitTimeUTC"] = now
        trades[idx]["PctReturn"] = f"{pct_return:.2f}"
        trades[idx]["ExitMessageTS"] = ts

        print(f"[EXIT] Closed {ticker} at {current_price:.4f}, PctReturn={pct_return:.2f}%")

    return trades


def main():
    client = get_client()
    trades = load_trades()

    # Look back over recent messages (last ~3 days)
    oldest_ts = (dt.datetime.utcnow() - dt.timedelta(days=3)).timestamp()

    try:
        result = client.conversations_history(
            channel=SLACK_CHANNEL_ID,
            oldest=str(oldest_ts),
            limit=200,
            inclusive=True,
        )
    except SlackApiError as e:
        print(f"[SLACK] Error fetching history: {e.response['error']}")
        return

    messages = result.get("messages", [])

    # Process messages from oldest to newest (so entries happen before exits)
    for m in reversed(messages):
        if is_entry_message(m):
            trades = sync_entries_from_message(m, trades)
        elif is_exit_message(m):
            trades = sync_exits_from_message(m, trades)

    save_trades(trades)
    print(f"[DONE] Synced trades. Total rows in {TRADES_FILE}: {len(trades)}")


if __name__ == "__main__":
    main()

