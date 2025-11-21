import datetime as dt
import math
import csv

import numpy as np
import pandas as pd
import yfinance as yf

from config import ACCOUNT_EQUITY, RISK_PER_TRADE_PCT, UNIVERSE_FILE


############################################################
# Shared helpers (similar to backtest_v9 / v10)
############################################################

def load_universe():
    try:
        with open(UNIVERSE_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Universe file '{UNIVERSE_FILE}' not found.")
        return []


def fetch(ticker, start, end):
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
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def get_market_regime(end_date: dt.date, lookback_days: int = 40):
    start = end_date - dt.timedelta(days=lookback_days)

    spy = fetch("SPY", start, end_date)
    qqq = fetch("QQQ", start, end_date)

    if spy is None or qqq is None or len(spy) < 20 or len(qqq) < 20:
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
    }


def get_relative_strength(stock_df: pd.DataFrame, spy_df: pd.DataFrame, as_of: dt.date, lookback: int = 5):
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


def score_ticker_v9(ticker: str, as_of: dt.date, df: pd.DataFrame, spy_df: pd.DataFrame):
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

    gap_pct = (last_close - prev_close) / prev_close
    if not (0.006 <= gap_pct <= 0.040):
        return None

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

    rs = get_relative_strength(df, spy_df, as_of, lookback=5)
    if rs is None or rs <= -0.01:
        return None

    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    proxy_vwap = (last_high + last_low + last_close) / 3.0

    if last_close < proxy_vwap:
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

    score += min(rs * 200.0, 10.0)

    return {
        "ticker": ticker,
        "as_of": as_of,
        "price": last_close,
        "gap_pct": gap_pct,
        "atr": atr,
        "score": score,
    }


def get_spy_trading_days(start, end):
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Could not get SPY history.")
    return list(df.index.date), df


def build_position_size(entry_price, atr, stop_mult):
    risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE_PCT
    stop_distance = stop_mult * atr
    if stop_distance <= 0:
        return None
    raw_qty = risk_dollars / stop_distance
    qty = max(1, math.floor(raw_qty))
    max_affordable_qty = math.floor(ACCOUNT_EQUITY / entry_price)
    if max_affordable_qty <= 0:
        return None
    return min(qty, max_affordable_qty)


def simulate_trade(signal_date, signal, df, trading_days, stop_mult, target_mult):
    try:
        idx = trading_days.index(signal_date)
    except ValueError:
        return None
    if idx + 1 >= len(trading_days):
        return None

    trade_date = trading_days[idx + 1]
    df_trade = df[df.index.date == trade_date]
    df_signal = df[df.index.date == signal_date]

    if df_trade.empty or df_signal.empty:
        return None

    prev_high = float(df_signal["High"].iloc[-1])

    open_px = float(df_trade["Open"].iloc[0])
    high_px = float(df_trade["High"].iloc[0])
    low_px = float(df_trade["Low"].iloc[0])
    close_px = float(df_trade["Close"].iloc[0])

    # ORB: open > prev_high or intraday high > prev_high
    if not (open_px > prev_high or high_px > prev_high):
        return None

    atr = signal["atr"]
    entry = open_px
    stop = entry - stop_mult * atr
    target = entry + target_mult * atr

    qty = build_position_size(entry, atr, stop_mult)
    if qty is None:
        return None

    if low_px <= stop and high_px >= target:
        exit_price = stop
    elif high_px >= target:
        exit_price = target
    elif low_px <= stop:
        exit_price = stop
    else:
        exit_price = close_px

    pnl_per_share = exit_price - entry
    pnl_dollars = pnl_per_share * qty
    pnl_pct = pnl_dollars / ACCOUNT_EQUITY * 100.0

    return pnl_dollars, pnl_pct


def run_single_backtest(stop_mult, target_mult, days_back=252, top_n=3):
    today = dt.date.today()
    history_buffer_days = 90
    start_date = today - dt.timedelta(days=days_back + history_buffer_days)
    end_date = today

    trading_days, spy_df = get_spy_trading_days(start_date, end_date)
    if len(trading_days) <= days_back + 1:
        signal_days = trading_days[:-1]
    else:
        signal_days = trading_days[-(days_back + 1):-1]

    universe = load_universe()
    if not universe:
        print("[ERROR] Universe empty.")
        return None

    all_data = {}
    for t in universe:
        df = fetch(t, start_date, end_date)
        if df is None or df.empty:
            continue
        all_data[t] = df

    records = []

    for d in signal_days:
        regime = get_market_regime(d)
        if not regime["allow_longs"]:
            continue

        day_signals = []
        for t, df in all_data.items():
            try:
                info = score_ticker_v9(t, d, df, spy_df)
            except Exception:
                continue
            if info is not None:
                day_signals.append(info)

        if not day_signals:
            continue

        day_signals.sort(key=lambda x: x["score"], reverse=True)
        selected = day_signals[:top_n]

        for s in selected:
            df_t = all_data[s["ticker"]]
            res = simulate_trade(d, s, df_t, trading_days, stop_mult, target_mult)
            if res is None:
                continue
            pnl_dollars, pnl_pct = res
            records.append((pnl_dollars, pnl_pct))

    if not records:
        return {
            "stop_mult": stop_mult,
            "target_mult": target_mult,
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    pnl_dollars_list = [r[0] for r in records]
    wins = [p for p in pnl_dollars_list if p > 0]
    losses = [p for p in pnl_dollars_list if p <= 0]

    total_pnl = sum(pnl_dollars_list)
    total_return_pct = sum(r[1] for r in records)
    win_rate = len(wins) / len(records) * 100.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    result = {
        "stop_mult": stop_mult,
        "target_mult": target_mult,
        "trades": len(records),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    return result


def main():
    stop_grid = [0.8, 1.0, 1.2]
    target_grid = [1.5, 2.0, 2.5]

    results = []

    print("[INFO] Running V11 optimizer (approx 1-year window)...")
    for s in stop_grid:
        for t in target_grid:
            print(f"\n[PARAM] STOP={s}, TARGET={t}")
            res = run_single_backtest(s, t, days_back=252, top_n=3)
            if res is None:
                continue
            results.append(res)
            print(
                f"Trades: {res['trades']}, Win: {res['win_rate']:.2f}%, "
                f"PnL: ${res['total_pnl']:.2f}, Return: {res['total_return_pct']:.2f}%"
            )

    if not results:
        print("[RESULT] No optimizer results (likely no trades).")
        return

    csv_file = "optimizer_v11_results.csv"
    fieldnames = [
        "stop_mult", "target_mult", "trades",
        "win_rate", "total_pnl", "total_return_pct",
        "avg_win", "avg_loss"
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[RESULT] Optimizer results written to {csv_file}")
    print("\n========== V11 OPTIMIZER SUMMARY ==========")
    for r in sorted(results, key=lambda x: x["total_return_pct"], reverse=True):
        print(
            f"STOP={r['stop_mult']}, TARGET={r['target_mult']} | "
            f"Trades={r['trades']}, Win={r['win_rate']:.2f}%, "
            f"Return={r['total_return_pct']:.2f}%"
        )
    print("===========================================")


if __name__ == "__main__":
    main()

