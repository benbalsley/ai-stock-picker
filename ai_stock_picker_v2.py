import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from config import SLACK_WEBHOOK_URL, UNIVERSE_FILE


############################################################
# Helpers
############################################################

def send_slack(message: str):
    """Send a Slack message to your Slack webhook."""
    if not SLACK_WEBHOOK_URL:
        print("[WARN] SLACK_WEBHOOK_URL not set, printing instead:\n", message)
        return
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        print("Slack response:", resp.status_code)
    except Exception as e:
        print("Slack error:", e)


def load_universe():
    """Load tickers from universe.txt (or whatever UNIVERSE_FILE is)."""
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Universe file '{UNIVERSE_FILE}' not found.")
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
# Market Regime & Filters
############################################################

def get_market_regime(end_date: dt.date, lookback_days: int = 40):
    """
    Use SPY & QQQ to determine if the market is supportive for LONGS.
    Returns dict with trend flags and a 'allow_longs' boolean.
    """
    start = end_date - dt.timedelta(days=lookback_days)

    spy = fetch("SPY", start, end_date)
    qqq = fetch("QQQ", start, end_date)

    if spy is None or qqq is None or len(spy) < 20 or len(qqq) < 20:
        print("[WARN] Could not get full SPY/QQQ data, defaulting to neutral regime.")
        return {"allow_longs": True}

    spy_close = spy["Close"]
    qqq_close = qqq["Close"]

    spy_ma20 = spy_close.rolling(20).mean().iloc[-1]
    qqq_ma20 = qqq_close.rolling(20).mean().iloc[-1]

    spy_last = spy_close.iloc[-1]
    qqq_last = qqq_close.iloc[-1]

    spy_ret = (spy_last - spy_close.iloc[-2]) / spy_close.iloc[-2]
    qqq_ret = (qqq_last - qqq_close.iloc[-2]) / qqq_close.iloc[-2]

    spy_up_trend = spy_last > spy_ma20
    qqq_up_trend = qqq_last > qqq_ma20

    # Basic low-vol / chop detection
    spy_range_pct = (spy["High"].iloc[-1] - spy["Low"].iloc[-1]) / spy_last
    qqq_range_pct = (qqq["High"].iloc[-1] - qqq["Low"].iloc[-1]) / qqq_last

    chop = (spy_range_pct < 0.002 and qqq_range_pct < 0.002)

    allow_longs = (
        spy_up_trend and qqq_up_trend and spy_ret > 0 and qqq_ret > 0 and not chop
    )

    return {
        "allow_longs": allow_longs,
        "spy_up_trend": spy_up_trend,
        "qqq_up_trend": qqq_up_trend,
        "spy_ret": spy_ret,
        "qqq_ret": qqq_ret,
        "chop": chop,
    }


############################################################
# Scoring V2
############################################################

def score_ticker_v2(ticker: str, as_of: dt.date, df: pd.DataFrame):
    """
    V2 institutional-style scoring for a single ticker as of 'as_of' date.
    Long-only, with:
      - Gap sweet spot (0.8%–3.5%)
      - Volume filter (avg > 1M)
      - Relative volume filter (>1.2 preferred)
      - Trend filter vs MA20
    """
    # Restrict to data up to as_of date
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

    # Gap %
    gap_pct = (last_close - prev_close) / prev_close

    # Sweet spot: 0.8%–3.5% gap up for longs
    if not (0.008 <= gap_pct <= 0.035):
        return None

    # Trend: require price > MA20 for long bias
    if last_close <= ma20:
        return None

    # Volume filters
    avg_vol20 = float(vol.rolling(20).mean().iloc[-1])
    today_vol = float(vol.iloc[-1])

    if avg_vol20 < 1_000_000:  # illiquid → skip
        return None

    rel_vol = today_vol / avg_vol20 if avg_vol20 > 0 else 0.0

    # Basic volatility normalization
    vol_score = (atr / last_close) * 100.0
    if vol_score < 0.5:
        return None

    # Scoring components
    score = 0.0

    # Gap size within sweet spot (peak preference around 1.5–2.5%)
    score += 40.0 * min(gap_pct * 100.0 / 2.0, 3.0)

    # Relative volume
    if rel_vol >= 2.0:
        score += 25.0
    elif rel_vol >= 1.5:
        score += 15.0
    elif rel_vol >= 1.2:
        score += 5.0
    else:
        score -= 10.0

    # Volatility: modest reward (not too low, not insane)
    if 1.0 <= vol_score <= 4.0:
        score += 10.0
    elif vol_score > 6.0:
        score -= 5.0

    # Trend strength (distance above MA20)
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
# Daily V2 Picker
############################################################

def pick_best_v2(as_of: dt.date | None = None):
    """
    V2 daily pick:
      - Market regime filter (SPY/QQQ)
      - Scans universe with V2 scoring
      - Returns best candidate or a reason for no trade
    """
    if as_of is None:
        as_of = dt.date.today()

    universe = load_universe()
    if not universe:
        return {"ticker": None, "reason": "Universe is empty."}

    # Market regime
    regime = get_market_regime(as_of)
    if not regime.get("allow_longs", True):
        return {
            "ticker": None,
            "reason": "Market regime not supportive for longs "
                      "(SPY/QQQ trend or chop filter).",
        }

    start = as_of - dt.timedelta(days=90)

    best = None
    best_info = None

    for t in universe:
        df = fetch(t, start, as_of + dt.timedelta(days=1))
        if df is None:
            continue
        try:
            info = score_ticker_v2(t, as_of, df)
        except Exception as e:
            print(f"[WARN] scoring error for {t}: {e}")
            continue
        if info is None:
            continue

        if best_info is None or info["score"] > best_info["score"]:
            best_info = info
            best = t

    if best_info is None:
        return {"ticker": None, "reason": "No candidate passed V2 filters."}

    return best_info


############################################################
# MAIN: Slack message for V2 (for now: manual / experimental)
############################################################

def main():
    today = dt.date.today()
    pick = pick_best_v2(today)

    if pick.get("ticker") is None:
        msg = (
            "🧪 V2 Model – No Play Today\n"
            f"Reason: {pick.get('reason', 'Unknown')}"
        )
        print(msg)
        send_slack(msg)
        return

    msg = (
        "🧪 V2 Model – Good Morning 🌅\n"
        f"*Ticker*: {pick['ticker']}\n"
        f"*As of*: {pick['as_of']}\n"
        f"*Price*: {pick['price']:.2f}\n"
        f"*Gap*: {pick['gap_pct']*100:.2f}%\n"
        f"*Rel Volume*: {pick['rel_vol']:.2f}x\n"
        f"*Vol Score*: {pick['vol_score']:.2f}\n"
        f"*Trend vs MA20*: {pick['trend_strength']*100:.2f}% above MA20\n"
        f"*ATR*: {pick['atr']:.2f}\n"
        f"*Score*: {pick['score']:.2f}\n\n"
        "Use this as an institutional-filtered candidate; "
        "entries/targets as per your current playbook."
    )

    print(msg)
    send_slack(msg)


if __name__ == "__main__":
    main()

