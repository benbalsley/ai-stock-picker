import datetime as dt
import math

import pandas as pd
import yfinance as yf
import requests

from config import SLACK_WEBHOOK_URL, ACCOUNT_EQUITY


POSITIONS_FILE = "positions_v9.csv"

# Threat thresholds
MAX_DRAWDOWN_PCT = -5.0   # threat if loss worse than -5%
BIG_MOVE_PCT = 4.0        # threat if today's move > 4% in either direction


def send_slack(msg: str):
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=10)
        if resp.status_code != 200:
            print(f"[SLACK] Error {resp.status_code}: {resp.text}")
        else:
            print("[SLACK] Message sent.")
    except Exception as e:
        print(f"[SLACK] Exception: {e}")


def load_positions():
    """
    Expected CSV schema (positions_v9.csv):

    Ticker,Direction,Entry,Stop,Target,Size,Notes
    AAPL,long,195,188,210,10,Core swing
    TSLA,short,250,265,220,5,Earnings play
    """
    try:
        df = pd.read_csv(POSITIONS_FILE)
    except FileNotFoundError:
        print(f"[INFO] Positions file '{POSITIONS_FILE}' not found.")
        return None
    except Exception as e:
        print(f"[ERROR] Reading positions file: {e}")
        return None

    required_cols = {"Ticker", "Direction", "Entry", "Stop", "Target", "Size"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] Positions file missing required columns: {required_cols}")
        return None

    return df


def latest_price_and_change(ticker: str):
    """
    Returns (last_close, day_change_pct) based on last 2 daily bars.
    """
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
    except Exception as e:
        print(f"[ERROR] Downloading {ticker}: {e}")
        return None, None

    if df is None or df.empty or len(df) < 2:
        print(f"[WARN] Not enough data for {ticker}")
        return None, None

    close = df["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    day_change_pct = (last - prev) / prev * 100.0
    return last, day_change_pct


def analyze_position(row):
    """
    Analyze a single position row and return a dict with status.
    """
    ticker = str(row["Ticker"]).strip().upper()
    direction = str(row["Direction"]).strip().lower()
    entry = float(row["Entry"])
    stop = float(row["Stop"])
    target = float(row["Target"])
    size = float(row["Size"])
    notes = str(row.get("Notes", "")).strip()

    if direction not in ("long", "short"):
        return None

    last_price, day_change_pct = latest_price_and_change(ticker)
    if last_price is None:
        return None

    if direction == "long":
        pnl_pct = (last_price - entry) / entry * 100.0
        stop_hit = last_price <= stop
        target_hit = last_price >= target
    else:  # short
        pnl_pct = (entry - last_price) / entry * 100.0
        stop_hit = last_price >= stop
        target_hit = last_price <= target

    threat = False
    threat_reasons = []

    if pnl_pct <= MAX_DRAWDOWN_PCT:
        threat = True
        threat_reasons.append(f"Drawdown {pnl_pct:.1f}%")

    if abs(day_change_pct) >= BIG_MOVE_PCT:
        threat = True
        direction_word = "up" if day_change_pct > 0 else "down"
        threat_reasons.append(f"Big move {day_change_pct:+.1f}% {direction_word} today")

    pos_notional = last_price * size
    risk_perc_of_acct = (abs(entry - stop) * size) / ACCOUNT_EQUITY * 100.0

    return {
        "ticker": ticker,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "size": size,
        "notes": notes,
        "last_price": last_price,
        "pnl_pct": pnl_pct,
        "day_change_pct": day_change_pct,
        "stop_hit": stop_hit,
        "target_hit": target_hit,
        "threat": threat,
        "threat_reasons": threat_reasons,
        "pos_notional": pos_notional,
        "risk_perc_of_acct": risk_perc_of_acct,
    }


def main():
    today = dt.date.today()

    positions_df = load_positions()
    if positions_df is None or positions_df.empty:
        send_slack(f"Risk monitor (V9) – {today}: No open positions found in positions_v9.csv.")
        return

    analyses = []
    for _, row in positions_df.iterrows():
        info = analyze_position(row)
        if info is not None:
            analyses.append(info)

    if not analyses:
        send_slack(f"Risk monitor (V9) – {today}: No analyzable positions.")
        return

    urgent = []
    warnings = []
    ok = []

    for a in analyses:
        if a["stop_hit"] or a["target_hit"]:
            urgent.append(a)
        elif a["threat"]:
            warnings.append(a)
        else:
            ok.append(a)

    lines = []
    lines.append(f"*Risk Monitor V9 – {today}*")
    lines.append("")

    if urgent:
        lines.append(":rotating_light: *URGENT – Stops/Targets Hit*")
        for a in urgent:
            line = (
                f"• {a['ticker']} ({a['direction']}) @ ${a['last_price']:.2f} – PnL: {a['pnl_pct']:+.1f}%\n"
                f"  • Stop: ${a['stop']:.2f}   Target: ${a['target']:.2f}   Size: {a['size']:.0f} "
                f"(risk ~{a['risk_perc_of_acct']:.1f}% of acct)"
            )
            if a["stop_hit"]:
                line += "  ➜ *STOP HIT*"
            if a["target_hit"]:
                line += "  ➜ *TARGET HIT*"
            if a["notes"]:
                line += f"\n  Notes: {a['notes']}"
            lines.append(line)
            lines.append("")
    else:
        lines.append("_No stops or targets hit._")
        lines.append("")

    if warnings:
        lines.append(":warning: *THREATENED POSITIONS*")
        for a in warnings:
            reasons = "; ".join(a["threat_reasons"])
            line = (
                f"• {a['ticker']} ({a['direction']}) @ ${a['last_price']:.2f} – PnL: {a['pnl_pct']:+.1f}%\n"
                f"  • Reasons: {reasons}\n"
                f"  • Stop: ${a['stop']:.2f}   Target: ${a['target']:.2f}   Size: {a['size']:.0f} "
                f"(risk ~{a['risk_perc_of_acct']:.1f}% of acct)"
            )
            if a["notes"]:
                line += f"\n  Notes: {a['notes']}"
            lines.append(line)
            lines.append("")
    else:
        lines.append("_No threatened positions based on current thresholds._")
        lines.append("")

    # You may or may not want this; optional.
    if ok:
        lines.append(":white_check_mark: *Stable positions*")
        for a in ok:
            line = (
                f"• {a['ticker']} ({a['direction']}) @ ${a['last_price']:.2f} – PnL: {a['pnl_pct']:+.1f}% "
                f"| ΔToday: {a['day_change_pct']:+.1f}%"
            )
            if a["notes"]:
                line += f"  | Notes: {a['notes']}"
            lines.append(line)
        lines.append("")

    msg = "\n".join(lines)
    print(msg)
    send_slack(msg)


if __name__ == "__main__":
    main()

