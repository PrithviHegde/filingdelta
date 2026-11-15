"""Find the earnings release that goes with each filing in the manifest.

10-Qs and 10-Ks usually come after the quarter's earnings release, sometimes the same
evening (MSFT, GOOGL, NVDA), sometimes weeks later (JPM, MS, WMT). The event study has to
know which, or it credits the filing with the earnings reaction.

Source: 8-Ks with Item 2.02 ("Results of Operations and Financial Condition"), the item
companies use to furnish the earnings press release. From SEC itself, with exact
acceptance times, rather than a price vendor's earnings calendar.

Rule: a filing's earnings release is the latest Item 2.02 8-K accepted after its quarter
ended and at or before the filing itself, at most MAX_GAP_DAYS earlier. "Latest" matters:
some companies file an earlier interim 2.02 (CVX 2026-04-09, before its 05-01 release).

Writes data/processed/earnings.csv: one row per manifest filing (amendments left out).

Run from the project root:
    python -m src.earnings
    python -m src.earnings --tickers MSFT   # print only, no write
"""

import argparse

import pandas as pd
from edgar import Company, set_identity

from src.config import DATA_DIR, ROOT, SEC_IDENTITY, TICKERS, EXTRA_CIKS
from src.ingest import MARKET_CLOSE

MANIFEST_PATH = DATA_DIR / "manifest.csv"
EARNINGS_PATH = DATA_DIR / "earnings.csv"

# Longest gap seen between release and filing is 36 days (MS 10-K). A 2.02 8-K older than
# this belongs to some other event; better to leave the filing unmatched and look at it.
MAX_GAP_DAYS = 60


def earnings_8ks(ticker, since):
    """Item 2.02 8-Ks for a ticker (all its CIKs) accepted on or after `since`, as a
    DataFrame with accession_no and acceptance_datetime (header time, US Eastern)."""
    ciks = [Company(ticker).cik] + EXTRA_CIKS.get(ticker, [])
    rows = []
    for cik in ciks:
        filings = Company(cik).get_filings(form="8-K")   # 8-K/A left out: amendments
        index = filings.to_pandas()
        keep = index["items"].fillna("").str.split(",").apply(lambda items: "2.02" in items) \
            & (pd.to_datetime(index["filing_date"]) >= since - pd.Timedelta(days=1))
        for accession in index.loc[keep, "accession_number"]:
            filing = filings.get(accession)
            # Header time, as in the manifest: the filings list can be hours off
            rows.append({"accession_no": accession,
                         "acceptance_datetime": filing.header.acceptance_datetime})
    return pd.DataFrame(rows, columns=["accession_no", "acceptance_datetime"])


def match(filings, releases):
    """One row per filing with the earnings release matched to it (or empty)."""
    rows = []
    for f in filings.itertuples():
        window = releases[(releases["acceptance_datetime"] > f.period_of_report)
                          & (releases["acceptance_datetime"] <= f.acceptance_datetime)
                          & (releases["acceptance_datetime"] >= f.acceptance_datetime
                             - pd.Timedelta(days=MAX_GAP_DAYS))]
        r = window.sort_values("acceptance_datetime").iloc[-1] if len(window) else None
        rows.append({
            "ticker": f.ticker, "form": f.form, "accession_no": f.accession_no,
            "period_of_report": f.period_of_report.date(),
            "acceptance_datetime": f.acceptance_datetime,
            "earnings_accession_no": r["accession_no"] if r is not None else None,
            "earnings_acceptance_datetime": r["acceptance_datetime"] if r is not None else None,
            "earnings_after_close": r["acceptance_datetime"].time() >= MARKET_CLOSE
                                    if r is not None else None,
            # Calendar days; the trading-day gap is computed in event_study.py
            "days_after_earnings": (f.acceptance_datetime - r["acceptance_datetime"]).days
                                   if r is not None else None,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="only these tickers: print, don't write")
    args = parser.parse_args()
    set_identity(SEC_IDENTITY)

    manifest = pd.read_csv(MANIFEST_PATH, parse_dates=["acceptance_datetime", "period_of_report"])
    manifest = manifest[~manifest["is_amendment"]]
    tickers = args.tickers or TICKERS

    out = []
    for ticker in tickers:
        filings = manifest[manifest["ticker"] == ticker].sort_values("period_of_report")
        releases = earnings_8ks(ticker, since=filings["period_of_report"].min())
        matched = match(filings, releases)
        out.append(matched)
        print(f"{ticker}: {len(releases)} item 2.02 8-Ks, "
              f"{matched['earnings_accession_no'].notna().sum()} of {len(matched)} filings matched, "
              f"days after earnings {matched['days_after_earnings'].tolist()}")
        for f in matched[matched["earnings_accession_no"].isna()].itertuples():
            print(f"  WARNING: no earnings release for {f.form} {f.period_of_report}")
    earnings = pd.concat(out, ignore_index=True)

    if args.tickers:
        print(earnings.to_string())
        print("\n--tickers given: earnings.csv not written")
    else:
        earnings.to_csv(EARNINGS_PATH, index=False)
        print(f"\nWrote {len(earnings)} rows to {EARNINGS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
