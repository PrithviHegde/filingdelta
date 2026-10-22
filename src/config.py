# Define paths

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # src/ → project root
DATA_DIR = ROOT / "data" / "processed"


# Company list for 10-K and 10-Q filings

TICKERS = ["JPM", "MS", "GOOGL", "MSFT", "NVDA", "XOM", "CVX", "COP", "WMT", "PG", "KO"]
FORMS = ["10-Q", "10-K"]
N_QUARTERS = 8
SEC_IDENTITY = "Prithvi Hegde prithvihegde123@gmail.com"