# Define paths

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # src/ → project root
DATA_DIR = ROOT / "data" / "processed"


# Company list for 10-K and 10-Q filings

TICKERS = ["JPM", "MS", "GOOGL", "MSFT", "NVDA", "XOM", "CVX", "COP", "WMT", "PG", "KO"]
FORMS = ["10-Q", "10-K"]
N_QUARTERS = 8
SEC_IDENTITY = "Prithvi Hegde prithvihegde123@gmail.com"

# Benchmarks for the event study: SPY for the market, plus one SPDR sector ETF per ticker.
# GOOGL is in Communication Services (XLC) under GICS since 2018, not Technology (XLK).
MARKET_ETF = "SPY"
SECTOR_ETF = {
    "JPM": "XLF", "MS": "XLF",
    "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLC",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    "WMT": "XLP", "PG": "XLP", "KO": "XLP",
}

# Extra SEC registrants whose filings belong to a ticker. XOM redomiciled to Texas on
# 2026-07-01: ExxonMobil Holdings Corp (CIK 2115436) became the successor registrant, and
# files the earnings 8-Ks from then on, while the Q2 2026 10-Q still came from the old
# Exxon Mobil Corp (CIK 34088), which edgartools' ticker lookup returns.
EXTRA_CIKS = {"XOM": [2115436]}
