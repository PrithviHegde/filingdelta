"""Download 10-Q / 10-K filings from EDGAR and save their MD&A sections as text.

Run from the project root:
    python -m src.ingest
"""

from pathlib import Path
from edgar import Company, set_identity
from src.config import SEC_IDENTITY, TICKERS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

# MD&A is Item 2 of Part I in a 10-Q, and Item 7 (Part II) in a 10-K
MDNA_SECTION = {
    "10-Q": "part_i_item_2",
    "10-K": "part_ii_item_7",
}


def get_10q_mdna(filing):
    """Return the MD&A text of a 10-Q, or None if the parser can't find it."""
    if filing.form != "10-Q":
        raise ValueError(f"Expected 10-Q, got {filing.form}")

    document = filing.obj().document
    if MDNA_SECTION["10-Q"] not in document.get_available_sec_sections():
        return None

    return document.get_sec_section(MDNA_SECTION["10-Q"])


def get_10k_mdna(filing):
    """Return the MD&A text of a 10-K, or None if the parser can't find it."""
    if filing.form != "10-K":
        raise ValueError(f"Expected 10-K, got {filing.form}")

    document = filing.obj().document
    if MDNA_SECTION["10-K"] not in document.get_available_sec_sections():
        return None

    return document.get_sec_section(MDNA_SECTION["10-K"])


def save_mdna(time_period, mdna, ticker):
    """Save MD&A text to data/processed/{ticker}/{period}_mdna.txt.

    Returns True if saved, False if there was no text to save.
    """
    if mdna is None:
        print(f"  WARNING: no MD&A found for {ticker} {time_period}, skipping")
        return False

    out_dir = DATA_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{time_period}_mdna.txt"
    out_path.write_text(mdna, encoding="utf-8")
    print(f"  Saved {len(mdna):,} chars to {out_path.relative_to(ROOT)}")
    return True


def ingest_ticker(ticker, n_10q=6, n_10k=2):
    """Fetch the last n_10q 10-Qs and n_10k 10-Ks for a ticker and save their MD&A.

    Returns a summary dict, e.g.
        {"ticker": ticker, "saved": 7, "skipped": 1} """


    company = Company(ticker)
    latest_10Q_filings = company.get_filings(form="10-Q").latest(n_10q)
    latest_10K_filings = company.get_filings(form="10-K").latest(n_10k)

    skipped, saved = 0, 0

    for tenQ in latest_10Q_filings:
        try:
            mdna = get_10q_mdna(tenQ)
            time_period = tenQ.period_of_report
            if save_mdna(time_period, mdna, ticker):
                saved +=1
            else:
                skipped +=1
        except Exception as e:
            print(f" ERROR {ticker} {tenQ.form} {tenQ.period_of_report}: {type(e).__name__}: {e}")
            skipped +=1

    for tenK in latest_10K_filings:
        try:
            mdna = get_10k_mdna(tenK)
            time_period = tenK.period_of_report
            if save_mdna(time_period, mdna, ticker):
                saved +=1
            else:
                skipped +=1
        except Exception as e:
            print(f" ERROR {ticker} {tenK.form} {tenK.period_of_report}: {type(e).__name__}: {e}")
            skipped +=1


    return {"ticker": ticker, "saved": saved, "skipped": skipped}


def main():
    set_identity(SEC_IDENTITY)

    summaries = []
    for ticker in TICKERS:
        print(f"\n{ticker}")
        summaries.append(ingest_ticker(ticker))

    print("\nSummary")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
