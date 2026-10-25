"""Download 10-Q / 10-K filings from EDGAR, save their MD&A sections as text,
and write a manifest with one row of metadata per filing.

Run from the project root:
    python -m src.ingest                  # download, extract and save MD&A, write manifest
    python -m src.ingest --manifest-only  # MD&A already saved: only fetch metadata, write manifest
"""

import argparse
from datetime import date, time

import pandas as pd
from edgar import Company, set_identity

from src.config import ROOT, DATA_DIR, SEC_IDENTITY, TICKERS

# MD&A is Item 2 of Part I in a 10-Q, and Item 7 (Part II) in a 10-K
MDNA_SECTION = {
    "10-Q": "part_i_item_2",
    "10-K": "part_ii_item_7",
}

# The manifest is the index of everything ingest.py produced: one row per filing, read by
# pairs.py (which filing to compare with which) and later by the event study (event dates).
# Fixed column order so the CSV layout doesn't depend on dict order or on which rows exist.
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MANIFEST_COLUMNS = [
    "ticker", "form", "is_amendment", "accession_no",
    "filing_date", "acceptance_datetime", "accepted_after_close",
    "period_of_report", "fiscal_year", "fiscal_quarter", "fiscal_year_end",
    "file_path",
]

# US market close. SEC acceptance times in the filing header are US Eastern.
MARKET_CLOSE = time(16, 0)


def get_10q_mdna(filing):
    """Return the MD&A text of a 10-Q (or 10-Q/A), or None if the parser can't find it."""
    if filing.form not in ("10-Q", "10-Q/A"):
        raise ValueError(f"Expected 10-Q or 10-Q/A, got {filing.form}")

    document = filing.obj().document
    if MDNA_SECTION["10-Q"] not in document.get_available_sec_sections():
        return None

    return document.get_sec_section(MDNA_SECTION["10-Q"])


def get_10k_mdna(filing):
    """Return the MD&A text of a 10-K (or 10-K/A), or None if the parser can't find it."""
    if filing.form not in ("10-K", "10-K/A"):
        raise ValueError(f"Expected 10-K or 10-K/A, got {filing.form}")

    document = filing.obj().document
    if MDNA_SECTION["10-K"] not in document.get_available_sec_sections():
        return None

    return document.get_sec_section(MDNA_SECTION["10-K"])


def mdna_path(ticker, time_period, amended=False):
    """Where the MD&A text for one filing lives: data/processed/{ticker}/{period}_mdna[_amended].txt

    Only builds the path, doesn't touch the disk. save_mdna writes to it, and
    --manifest-only uses it to check whether a file from an earlier run exists, so the
    naming rule is defined once and both modes always agree on where a file should be.
    """
    suffix = "_amended" if amended else ""
    return DATA_DIR / ticker / f"{time_period}_mdna{suffix}.txt"


def save_mdna(time_period, mdna, ticker, amended=False):
    """Save MD&A text to data/processed/{ticker}/{period}_mdna.txt.

    Amendments (10-Q/A, 10-K/A) cover the same period as the original filing,
    so they go to {period}_mdna_amended.txt instead of overwriting it.

    Returns True if saved, False if there was no text to save.
    """
    if mdna is None:
        print(f"  WARNING: no MD&A found for {ticker} {time_period}, skipping")
        return False

    out_path = mdna_path(ticker, time_period, amended)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mdna, encoding="utf-8")
    print(f"  Saved {len(mdna):,} chars to {out_path.relative_to(ROOT)}")
    return True


def fiscal_quarter(period_of_report, fiscal_year_end="1231"):
    """Map a period end date to (fiscal_year, fiscal_quarter), e.g. ("2026-03-31", "0630") -> (2026, 3).

    fiscal_year_end is SEC's MMDD string ("0630" for MSFT, "0131" for WMT).
    The fiscal year is named after the calendar year it ends in, as companies do:
    MSFT's quarter ending 2025-09-30 is Q1 of fiscal 2026. Quarter 4 is the 10-K.

    52/53-week calendars (NVDA, KO) end quarters on a weekday near the month end, so a
    quarter can end a few days into the next month (KO's Q2 2026 ended 2026-07-03).
    Dates in the first week of a month are therefore counted as the previous month.
    (Assumes fiscal_year_end itself is a late-month date, which holds for all 11 tickers.)

    Why not calendar quarters: MSFT, NVDA and WMT have non-December years, so calendar
    labels would pair the wrong filings in pairs.py (e.g. MSFT's Q1 with someone's Q3).

    Raises ValueError if the date isn't a multiple of 3 months from the fiscal year end,
    i.e. it can't be a quarter end, rather than silently guessing a quarter.
    """
    d = date.fromisoformat(str(period_of_report)[:10])  # accepts "2026-03-31" or a date/datetime
    fy_end_month = int(str(fiscal_year_end)[:2])        # "0630" -> 6

    # Snap 52/53-week dates back to the month the quarter really belongs to (2026-07-03 -> June)
    year, month = d.year, d.month
    if d.day <= 7:
        month -= 1
        if month == 0:
            year, month = year - 1, 12

    # Quarter ends are 3, 6, 9 or 12 months after the fiscal year end month.
    # E.g. MSFT (June year end), March: (3 - 6) % 12 = 9 months -> Q3.
    months_since_fy_end = (month - fy_end_month) % 12
    if months_since_fy_end % 3 != 0:
        raise ValueError(f"{period_of_report} is not a quarter end for fiscal year end {fiscal_year_end}")

    # 3 -> Q1, 6 -> Q2, 9 -> Q3; 0 months means the year-end month itself, which is Q4
    quarter = months_since_fy_end // 3 or 4
    # Months up to the year-end month belong to the fiscal year ending this calendar year;
    # later months to the one ending next year (MSFT Sep 2025 -> FY2026)
    fiscal_year = year if month <= fy_end_month else year + 1
    return fiscal_year, quarter


def filing_record(filing, ticker, fiscal_year_end, file_path):
    """One manifest row (a dict with the MANIFEST_COLUMNS keys) for a filing.

    file_path is the saved MD&A file, or None if there isn't one (parser found no MD&A).
    The row is still written in that case so the manifest shows every filing we fetched,
    including the failures, instead of them silently disappearing.

    Acceptance time comes from the filing's own SGML header (US Eastern time). The
    acceptance_datetime on the filings list can be hours off for some filers, and the
    4pm cutoff decides which trading day is the event day, so it needs to be exact.
    """
    acceptance = filing.header.acceptance_datetime
    period = str(filing.period_of_report)
    fiscal_year, quarter = fiscal_quarter(period, fiscal_year_end)

    # Store "10-Q"/"10-K" plus an is_amendment flag rather than "10-Q/A", so pairs.py can
    # filter amendments with one boolean instead of string matching.
    base_form = filing.form.removesuffix("/A")

    # Sanity check on fiscal_quarter(): a 10-K must land on Q4 and a 10-Q never should.
    # A mismatch means the fiscal-year logic is wrong for this company. Warn rather than
    # stop so one odd filing doesn't kill the run; the warning is what to look for in the log.
    expected_q4 = base_form == "10-K"
    if (quarter == 4) != expected_q4:
        print(f"  WARNING: {ticker} {filing.form} {period} maps to fiscal Q{quarter}, check fiscal_quarter()")

    return {
        "ticker": ticker,
        "form": base_form,
        "is_amendment": filing.form.endswith("/A"),
        "accession_no": filing.accession_no,
        "filing_date": str(filing.filing_date),
        # Left empty (not guessed) if the header has no acceptance time; check for these
        "acceptance_datetime": acceptance.isoformat(sep=" ") if acceptance else None,
        # True -> the market had already closed, so the event day is the next trading day
        "accepted_after_close": acceptance.time() >= MARKET_CLOSE if acceptance else None,
        "period_of_report": period,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": quarter,
        "fiscal_year_end": fiscal_year_end,
        "file_path": str(file_path.relative_to(ROOT)) if file_path else None,
    }


def ingest_ticker(ticker, n_10q=6, n_10k=2, manifest_only=False):
    """Fetch the last n_10q 10-Qs and n_10k 10-Ks for a ticker, save their MD&A, and
    return a manifest row for each.

    With manifest_only=True, nothing is downloaded or parsed beyond each filing's header:
    rows point to MD&A files already on disk, and file_path is empty where none exists.

    Returns (summary, rows), e.g. ({"ticker": ticker, "saved": 7, "skipped": 1}, [...])
    """
    company = Company(ticker)
    # SEC's MMDD fiscal year end ("0630" for MSFT). Fall back to December if it's missing,
    # which filing_record's Q4 check would then flag for non-December companies.
    fiscal_year_end = company.fiscal_year_end or "1231"
    # 10-Qs and 10-Ks go through one loop; get_filings(form=...) also returns amendments
    filings = list(company.get_filings(form="10-Q").head(n_10q)) + \
              list(company.get_filings(form="10-K").head(n_10k))

    skipped, saved = 0, 0
    rows = []

    for filing in filings:
        period = filing.period_of_report
        amended = filing.form.endswith("/A")
        try:
            if manifest_only:
                # Reuse MD&A from an earlier run: only check the file is there
                path = mdna_path(ticker, period, amended)
                ok = path.exists()
            else:
                # Pick the extractor by form, then parse the filing and save its MD&A
                get_mdna = get_10q_mdna if filing.form.startswith("10-Q") else get_10k_mdna
                ok = save_mdna(period, get_mdna(filing), ticker, amended=amended)
                path = mdna_path(ticker, period, amended)

            # Build the row before counting: if filing_record raises (e.g. fiscal_quarter
            # rejects the date), the except below counts the filing once, as skipped,
            # instead of it being counted as both saved and skipped.
            rows.append(filing_record(filing, ticker, fiscal_year_end, path if ok else None))
            # ok is a bool, so this adds 1 to exactly one of the two counters
            saved += ok
            skipped += not ok
        except Exception as e:
            # One bad filing (parser error, network hiccup) shouldn't stop the other tickers.
            # The filing gets no manifest row; the ERROR line in the log is the only trace.
            # (If the error came from filing_record, its MD&A file may already be on disk.)
            print(f" ERROR {ticker} {filing.form} {period}: {type(e).__name__}: {e}")
            skipped += 1

    return {"ticker": ticker, "saved": saved, "skipped": skipped}, rows


def write_manifest(rows):
    """Write all tickers' rows to data/processed/manifest.csv and return them as a DataFrame.

    Overwrites the whole file each run, so the manifest always matches the latest run
    (a --tickers MSFT run leaves a manifest with only MSFT in it).
    Sorted by ticker, then period, with an amendment right after its original filing,
    so the CSV is stable across runs and easy to read by eye.
    """
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(["ticker", "period_of_report", "is_amendment"]).reset_index(drop=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nWrote {len(manifest)} rows to {MANIFEST_PATH.relative_to(ROOT)}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest-only", action="store_true",
                        help="skip MD&A download; only fetch filing metadata and write the manifest")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="default: all tickers in config")
    args = parser.parse_args()

    set_identity(SEC_IDENTITY)

    summaries, rows = [], []
    for ticker in args.tickers:
        print(f"\n{ticker}")
        summary, ticker_rows = ingest_ticker(ticker, manifest_only=args.manifest_only)
        summaries.append(summary)
        rows.extend(ticker_rows)

    write_manifest(rows)

    print("\nSummary")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
