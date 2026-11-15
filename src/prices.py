"""Download daily prices for the tickers and their benchmark ETFs.

Writes data/processed/prices.csv: one row per trading day, one column per symbol, holding
the dividend- and split-adjusted close. Returns are computed from it in event_study.py.

Adjusted (not raw) closes, so a dividend or split doesn't show up as a price drop on its
ex-date. The start date leaves room for the event study's estimation window, which begins
250 trading days (about a year) before the earliest filing.

Run from the project root:
    python -m src.prices
"""

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, ROOT, TICKERS, MARKET_ETF, SECTOR_ETF

PRICES_PATH = DATA_DIR / "prices.csv"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Trading days per calendar year is ~252, so 250 trading days back from the earliest
# filing is ~363 calendar days. 500 calendar days leaves a margin for holidays.
LOOKBACK_CALENDAR_DAYS = 500


def symbols():
    """Tickers, then the market ETF, then the sector ETFs (each once)."""
    return TICKERS + [MARKET_ETF] + sorted(set(SECTOR_ETF.values()))


def download(start, end):
    """Adjusted closes, one column per symbol. end is exclusive (yfinance convention)."""
    data = yf.download(symbols(), start=start, end=end, auto_adjust=True, progress=False)
    closes = data["Close"][symbols()]
    closes.index = pd.to_datetime(closes.index).date
    closes.index.name = "date"
    return closes


def check(closes):
    """Print coverage and stop on gaps: a missing price would silently drop event days."""
    print(f"{len(closes)} trading days, {closes.index[0]} to {closes.index[-1]}, "
          f"{closes.shape[1]} symbols")
    missing = closes.isna().sum()
    if missing.any():
        raise ValueError(f"missing prices:\n{missing[missing > 0]}")
    # A one-day move this large in an adjusted mega-cap series is more likely a data
    # error (unadjusted split) than a real return; print it to look at, don't drop it.
    returns = closes.pct_change().iloc[1:]
    stacked = returns.stack()
    for (day, symbol), r in stacked[stacked.abs() > 0.25].items():
        print(f"  check: {symbol} {day} return {r:+.1%}")


def main():
    manifest = pd.read_csv(MANIFEST_PATH, parse_dates=["acceptance_datetime"])
    start = manifest["acceptance_datetime"].min() - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)
    end = pd.Timestamp.today().normalize()   # exclusive: today's bar may be incomplete
    closes = download(start.date(), end.date())
    check(closes)
    closes.to_csv(PRICES_PATH)
    print(f"Wrote {PRICES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
