"""Abnormal returns around each filing and its earnings release (market model).

For every filing in earnings.csv, and separately for its earnings release:
  1. event_day()      the first trading day the market could react: the acceptance day,
                      or the next trading day if accepted at/after the 4pm ET close
  2. market_model()   OLS of the stock's daily return on a benchmark's, over trading days
                      -250 to -30 before the event: alpha, beta, residual sigma
  3. abnormal returns AR_t = R_t - (alpha + beta * R_benchmark_t), summed over each CAR
                      window; SCAR = CAR / (sigma * sqrt(window length)), a rough t-stat

Benchmarks: SPY ("market", primary) and the ticker's SPDR sector ETF ("sector", robustness).
The sector model understates abnormal returns for stocks that are a large share of their
own ETF (MSFT and NVDA in XLK, XOM in XLE, GOOGL in XLC): part of the stock's own move is
in the benchmark, so it is subtracted out.

The estimation window ends 30 days before the event so the event itself (and the run-up
to it) can't leak into alpha and beta. It still contains earlier quarters' earnings days,
which widen sigma a little; that is the standard setup and leaves alpha / beta unbiased.

Writes data/processed/events.csv in long format: one row per filing x event (filing /
earnings) x benchmark x window. trading_days_after_earnings and earnings_in_window say
whether a filing's CAR also contains the earnings reaction (always when filed the same
day, as MSFT, GOOGL, NVDA do).

Run from the project root (needs prices.csv and earnings.csv):
    python -m src.event_study
    python -m src.event_study --tickers MSFT   # print only, no write
"""

import argparse

import numpy as np
import pandas as pd

from src.config import DATA_DIR, ROOT, MARKET_ETF, SECTOR_ETF
from src.ingest import MARKET_CLOSE

PRICES_PATH = DATA_DIR / "prices.csv"
EARNINGS_PATH = DATA_DIR / "earnings.csv"
EVENTS_PATH = DATA_DIR / "events.csv"

ESTIMATION_WINDOW = (-250, -30)   # trading days relative to the event day, inclusive
# [0,+1]: day the market could first react, plus one for slow digestion of a long filing.
# [-1,+1]: also catches leaks or a filing accepted just before an early close.
# [0,+5]: a week, for slower reactions to text that takes time to read.
CAR_WINDOWS = [(0, 1), (-1, 1), (0, 5)]
# Fewer estimation days than this (e.g. a trading halt) and the model isn't trusted
MIN_ESTIMATION_DAYS = 200


def event_day(acceptance, trading_days):
    """Index into trading_days of the first day the market could react to a filing.

    Accepted before 4pm ET on a trading day: that day (pre-market filings included).
    At/after 4pm, or on a weekend / market holiday (EDGAR is open on some, e.g. Good
    Friday): the next trading day.
    """
    day = acceptance.date()
    after_close = acceptance.time() >= MARKET_CLOSE
    side = "right" if after_close else "left"
    return int(np.searchsorted(trading_days, day, side=side))


def market_model(stock, benchmark):
    """OLS stock = alpha + beta * benchmark. Returns alpha, beta, residual sigma, n."""
    beta, alpha = np.polyfit(benchmark, stock, 1)
    residuals = stock - (alpha + beta * benchmark)
    sigma = residuals.std(ddof=2)   # two parameters estimated
    return alpha, beta, sigma, len(stock)


def event_rows(base, returns, ticker, day_idx, event, earnings_idx):
    """All benchmark x window rows for one event of one filing."""
    rows = []
    lo, hi = ESTIMATION_WINDOW
    if day_idx + lo < 0 or day_idx + max(b for _, b in CAR_WINDOWS) >= len(returns):
        raise ValueError(f"{ticker} {base['accession_no']} {event}: price history too short")
    est = returns.iloc[day_idx + lo: day_idx + hi + 1]
    for name, symbol in [("market", MARKET_ETF), ("sector", SECTOR_ETF[ticker])]:
        sample = est[[ticker, symbol]].dropna()
        if len(sample) < MIN_ESTIMATION_DAYS:
            raise ValueError(f"{ticker} {base['accession_no']}: {len(sample)} estimation days")
        alpha, beta, sigma, n = market_model(sample[ticker].to_numpy(), sample[symbol].to_numpy())
        for a, b in CAR_WINDOWS:
            win = returns.iloc[day_idx + a: day_idx + b + 1]
            ar = win[ticker] - (alpha + beta * win[symbol])
            car = float(ar.sum())
            rows.append({
                **base, "event": event, "event_day": returns.index[day_idx],
                "benchmark": name, "benchmark_symbol": symbol, "window": f"[{a:+d},{b:+d}]",
                "car": car, "scar": car / (sigma * np.sqrt(b - a + 1)),
                "alpha": alpha, "beta": beta, "sigma": sigma, "n_estimation": n,
                # Does this window contain the earnings release's event day?
                "earnings_in_window": day_idx + a <= earnings_idx <= day_idx + b,
            })
    return rows


def run(earnings, prices):
    returns = prices.pct_change().iloc[1:]
    trading_days = returns.index.to_numpy()
    rows = []
    for f in earnings.itertuples():
        filing_idx = event_day(f.acceptance_datetime, trading_days)
        earnings_idx = event_day(f.earnings_acceptance_datetime, trading_days)
        base = {"ticker": f.ticker, "form": f.form, "accession_no": f.accession_no,
                "period_of_report": f.period_of_report,
                "trading_days_after_earnings": filing_idx - earnings_idx}
        rows += event_rows(base, returns, f.ticker, filing_idx, "filing", earnings_idx)
        rows += event_rows(base, returns, f.ticker, earnings_idx, "earnings", earnings_idx)
    return pd.DataFrame(rows)


def print_report(events):
    """Event-day timing per ticker, then mean CAR / mean |CAR| per event and window."""
    filing = events[(events["event"] == "filing") & (events["benchmark"] == "market")
                    & (events["window"] == "[+0,+1]")]
    print("Trading days from earnings release to filing (per filing):")
    for ticker, g in filing.groupby("ticker", sort=False):
        print(f"  {ticker:6s} {g['trading_days_after_earnings'].tolist()}")

    summary = events.groupby(["benchmark", "event", "window"]).agg(
        n=("car", "size"), mean_car=("car", "mean"),
        mean_abs_car=("car", lambda c: c.abs().mean()),
        share_abs_scar_gt_2=("scar", lambda s: (s.abs() > 2).mean()),
        earnings_in_window=("earnings_in_window", "mean"))
    print("\n" + summary.round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="only these tickers: print, don't write")
    args = parser.parse_args()

    prices = pd.read_csv(PRICES_PATH, index_col="date", parse_dates=["date"])
    prices.index = prices.index.date
    earnings = pd.read_csv(EARNINGS_PATH, parse_dates=["acceptance_datetime",
                                                       "earnings_acceptance_datetime"])
    if earnings["earnings_acceptance_datetime"].isna().any():
        raise ValueError("filings without an earnings release in earnings.csv: check them first")
    if args.tickers:
        earnings = earnings[earnings["ticker"].isin(args.tickers)]

    events = run(earnings, prices)
    print_report(events)

    if args.tickers:
        print("\n--tickers given: events.csv not written")
    else:
        events.to_csv(EVENTS_PATH, index=False)
        print(f"\nWrote {len(events)} rows to {EVENTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
