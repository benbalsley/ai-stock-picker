import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import requests
import os
import json

from config import SLACK_WEBHOOK_URL


###############################################################
# Utility Functions
###############################################################

def send_slack(message: str):
    """Send a Slack message to webhook."""
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        print("Slack response:", resp.status_code)
    except Exception as e:
        print("Slack error:", e)


def load_universe():
    """Load tickers from universe.txt."""
    with open("universe.txt", "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def fetch(ticker, days=60):
    """Download recent price data."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)
    return df


def premarket_volume(ticker):
    """Get pre-market volume."""
    try:
        data = yf.Ticker(ticker).history(period="1d", prepost=True)
        pm = data.between_time("04:00", "09:30")["Volume"].sum()
        return pm
    except:
        return 0


def is_market_chop():
    """Market regime filter using SPY, QQQ, VIX."""
    spy = fetch("SPY", 10)
    qqq = fetch("QQQ", 10)

    spy_range = (spy["High"] - spy["Low"]).iloc[-1]
    qqq_range = (qqq["High"] - qqq["Low"]).iloc[-1]

    spy_close = spy["Close"].iloc[-1]
    qqq_close = qqq["Close"].iloc[-1]

    spy_atr = spy["Close"].rolling(14).std().iloc[-1]
    qqq_atr = qqq["Close"].rolling(14).std().iloc[-1]

    # Conditions for "Chop / No Trend"
    if spy_atr < 0.4 or qqq_atr < 0.5:
        return True
    if abs(spy_range / spy_close) < 0.002:   # <0.2%
        return True
    if abs(qqq_range / qqq_close) < 0.002:
        return True

    return False


def relative_volume(df):
    """Compute relative volume = current volume / avg volume."""
    today_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
    if avg_vol == 0:
        return 0
    return today_vol / avg_vol


###############################################################
# Scoring Function
###############################################################

def score_ticker(ticker):
    """Compute score for the daily pick with volume + regime filters."""
    df = fetch(ticker)
    if df is None or len(df) < 25:
        return None

    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]

    # Gap %
    gap = (last - prev) / prev

    # Trend: MA20
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = last - ma20

    # Relative Volume
    rv = relative_volume(df)

    # Pre-market Volume
    pmv = premarket_volume(ticker)

    score = 0

    # Gap scoring
    if gap > 0:
        score += min(3, gap * 50)   # Up gap
    else:
        score += min(3, abs(gap) * 50)

    # Trend scoring
    if trend > 0:
        score += 2
    else:
        score -= 1

    # Relative Volume scoring
    if rv > 1.5:
        score += 3
    elif rv > 1:
        score += 1
    else:
        score -= 2

    # Pre-market volume scoring
    if pmv > 300000:
        score += 2
    elif pmv > 100000:
        score += 1
    else:
        score -= 1

    return {
        "ticker": ticker,
        "gap": round(gap * 100, 2),
        "trend": trend,
        "rv": rv,
        "pmv": pmv,
        "score": score,
        "last": last
    }


###############################################################
# Main Pick Function
###############################################################

def pick_best():
    """Pick today's best ticker, skipping chop conditions."""
    universe = load_universe()

    if is_market_chop():
        return {"ticker": None, "reason": "Market in chop regime — no play today"}

    scored = []
    for t in universe:
        try:
            info = score_ticker(t)
            if info:
                scored.append(info)
        except:
            pass

    if not scored:
        return {"ticker": None, "reason": "No valid tickers after scoring"}

    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted[0]


###############################################################
# MAIN EXECUTION
###############################################################

def main():
    pick = pick_best()

    if pick["ticker"] is None:
        msg = f"Good morning 🌅\nNo play today.\nReason: {pick['reason']}"
        send_slack(msg)
        print(msg)
        return

    msg = (
        f"Good morning 🌅\n"
        f"**Today's Play: {pick['ticker']}**\n"
        f"Gap: {pick['gap']}%\n"
        f"Trend vs MA20: {pick['trend']:.2f}\n"
        f"Relative Volume: {pick['rv']:.2f}\n"
        f"Pre-market Vol: {pick['pmv']}\n"
        f"Score: {pick['score']}\n\n"
        f"📈 Entry Zone: Use discretion with volatility.\n"
    )

    send_slack(msg)
    print(msg)


if __name__ == "__main__":
    main()

