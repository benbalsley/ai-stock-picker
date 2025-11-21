import datetime as dt
import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from config import SLACK_WEBHOOK_URL, UNIVERSE_FILE, ACCOUNT_EQUITY, RISK_PER_TRADE_PCT


############################################################
# Exit parameters — from V9 (optimal so far)
############################################################

STOP_MULT = 1.0
TARGET_MULT = 2.0


############################################################
# Load Universe
############################################################

def load_universe():
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [x.strip() for x in f.readlines() if x.strip()]
    except FileNotFoundError:
        return []


############################################################
# Data Helpers
############################################################

def fetch(ticker, start, end):
    df = yf.download(
        ticker, start=start, end=end,
        progress=False, auto_adjust=False
    )
    if df is None or df.empty:
        return None
    return df.dropna()


def calc_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


############################################################
# Market Regime (SPY/QQQ)
############################################################

def get_market_regime(as_of):
    start = as_of - dt.timedelta(days=40)

    spy = fetch("SPY", start, as_of)
    qqq = fetch("QQQ", start, as_of)

    if spy is None or qqq is None:
        return {"allow_longs": True}

    spy_c = spy["Close"]
    qqq_c = qqq["Close"]

    spy_up = spy_c.iloc[-1] > spy_c.rolling(20).mean().iloc[-1]
    qqq_up = qqq_c.iloc[-1] > qqq_c.rolling(20).mean().iloc[-1]

    spy_ret = (spy_c.iloc[-1] - spy_c.iloc[-2]) / spy_c.iloc[-2]
    qqq_ret = (qqq_c.iloc[-1] - qqq_c.iloc[-2]) / qqq_c.iloc[-2]

    spy_rng = (spy["High"].iloc[-1] - spy["Low"].iloc[-1]) / spy_c.iloc[-1]
    qqq_rng = (qqq["High"].iloc[-1] - qqq["Low"].iloc[-1]) / qqq_c.iloc[-1]

    chop = spy_rng < 0.002 and qqq_rng < 0.002

    allow = spy_up and qqq_up and spy_ret > 0 and qqq_ret > 0 and not chop

    return {"allow_longs": allow}


############################################################
# Relative Strength vs SPY
############################################################

def get_rs(stock_df, spy_df, as_of, lookback=5):
    s = stock_df[stock_df.index.date <= as_of]
    p = spy_df[spy_df.index.date <= as_of]

    if len(s) < lookback + 2 or len(p) < lookback + 2:
        return None

    s_close = s["Close"]
    p_close = p["Close"]

    s_ret = (s_close.iloc[-1] - s_close.iloc[-lookback - 1]) / s_close.iloc[-lookback - 1]
    p_ret = (p_close.iloc[-1] - p_close.iloc[-lookback - 1]) / p_close.iloc[-lookback - 1]

    return s_ret - p_ret


############################################################
# Scoring (V9)
############################################################

def score_ticker(ticker, as_of, df, spy_df):
    df_up = df[df.index.date <= as_of]
    if len(df_up) < 30:
        return None

    close = df_up["Close"]
    high = df_up["High"]
    low = df_up["Low"]
    vol = df_up["Volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    ma20 = float(close.rolling(20).mean().iloc[-1])

    atr = float(calc_atr(df_up).iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return None

    gap_pct = (last_close - prev_close) / prev_close
    if not (0.006 <= gap_pct <= 0.040):
        return None

    if last_close <= ma20:
        return None

    avg_vol20 = float(vol.rolling(20).mean().iloc[-1])
    today_vol = float(vol.iloc[-1])
    if avg_vol20 < 1_000_000:
        return None

    rel_vol = today_vol / avg_vol20 if avg_vol20 else 0

    vol_score = (atr / last_close) * 100
    if vol_score < 0.5:
        return None

    rs = get_rs(df, spy_df, as_of, 5)
    if rs is None or rs <= -0.01:
        return None

    proxy_vwap = (float(high.iloc[-1]) + float(low.iloc[-1]) + last_close) / 3.0
    if last_close < proxy_vwap:
        return None

    score = 0.0

    score += 40 * min(gap_pct * 100 / 2, 3)
    if rel_vol >= 2: score += 25
    elif rel_vol >= 1.5: score += 15
    elif rel_vol >= 1.2: score += 5
    else: score -= 10

    if 1 <= vol_score <= 4:
        score += 10
    elif vol_score > 6:
        score -= 5

    trend_strength = (last_close - ma20) / ma20
    score += min(trend_strength * 200, 10)

    score += min(rs * 200, 10)

    return {
        "ticker": ticker,
        "score": score,
        "gap_pct": gap_pct,
        "atr": atr,
        "price": last_close
    }


############################################################
# Slack Notification
############################################################

def slack(msg):
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": msg})
    except Exception:
        pass


############################################################
# MAIN PICK LOGIC
############################################################

def main():
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    if today.weekday() >= 5:
        return  # weekend

    regime = get_market_regime(yesterday)
    if not regime["allow_longs"]:
        slack("Market conditions weak — no trades today.")
        return

    universe = load_universe()
    if not universe:
        slack("Universe is empty.")
        return

    spy_df = fetch("SPY", yesterday - dt.timedelta(days=40), yesterday)

    signals = []

    for t in universe:
        df = fetch(t, yesterday - dt.timedelta(days=60), yesterday)
        if df is None:
            continue

        info = score_ticker(t, yesterday, df, spy_df)
        if info:
            signals.append(info)

    if not signals:
        slack("No qualifying signals today.")
        return

    signals.sort(key=lambda x: x["score"], reverse=True)
    picks = signals[:3]

    msg = "*Good Morning — Top Trading Setups Today (V9)*\n\n"

    for i, p in enumerate(picks, 1):
        msg += (
            f"*{i}. {p['ticker']}*\n"
            f"Price: ${p['price']:.2f}\n"
            f"Gap: {p['gap_pct']*100:.2f}%\n"
            f"ATR: {p['atr']:.2f}\n"
            f"Score: {p['score']:.2f}\n\n"
        )

    slack(msg)


if __name__ == "__main__":
    main()

