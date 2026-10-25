"""Flag extracted MD&A sections that look wrong, so they can be checked by eye.

edgartools can "succeed" and still return the wrong text (notebook 01: BAC's share
repurchase table instead of MD&A). Saved is not correct, so every file gets three checks:
  too short    fewer than MIN_WORDS words: extraction probably cut off or got a stub
  too long     more than MAX_MEDIAN_MULTIPLE x the median: probably ran into other sections
  no heading   first HEADING_CHARS characters don't read like the start of MD&A

Adds word_count, median_words, valid and flag_reason to data/processed/manifest.csv.
Nothing is deleted: flagged filings stay in the manifest so each flag can be eyeballed.
Needs no SEC access; rerun after every `python -m src.ingest` (which rewrites the manifest).

Run from the project root:
    python -m src.validate                  # validate all filings, write the manifest
    python -m src.validate --tickers MSFT   # dry run on some tickers: print only, no write
    python -m src.validate --selftest       # check the three check functions on toy inputs
"""

import argparse

import pandas as pd

from src.config import ROOT, DATA_DIR

MANIFEST_PATH = DATA_DIR / "manifest.csv"
VALIDATION_COLUMNS = ["word_count", "median_words", "valid", "flag_reason", "review_note"]

# Flags checked by hand and found to be fine, keyed by accession number (unique per filing,
# unlike ticker + period, which an amendment shares). The flag stays in flag_reason so the
# record shows what fired; valid is True because the text is usable as is.
# Add an entry only after reading the file, and say what was checked.
REVIEWED_OK = {
    "0000019617-25-000270": "JPM 10-K 2024-12-31: real MD&A; heading at char ~5,100 after "
                            "~760 words of financial highlights (JPM 10-K layout). Reviewed 2026-10-24.",
    "0001628280-26-008131": "JPM 10-K 2025-12-31: real MD&A; heading at char ~5,650 after "
                            "~840 words of financial highlights (JPM 10-K layout). Reviewed 2026-10-24.",
}

# Thresholds from docs/plan.md. Starting points: adjust after looking at the data (Step 1).
# Shortest real MD&A observed 2026-10-24 is 4,407 words (NVDA 10-Q 2026-04-26); 5,000
# flagged three genuine filings. 3,000 sits well below that and still catches stubs.
MIN_WORDS = 3_000
# Too-long limits, as a multiple of the ticker's 10-Q median word count.
# Observed 2026-10-24 (11 tickers): 10-Qs are at most 1.19x their median (COP);
# 10-Ks are 0.9x (MSFT) to 3.9x (XOM), most 1.3-2.2x. Each limit sits well above
# its observed maximum, so only an extraction that ran into other sections trips it.
MAX_MULTIPLE_10Q = 2
MAX_MULTIPLE_10K = 5
HEADING_CHARS = 500


# ---------------------------------------------------------------------------
# Checks: Prithvi writes these. Each returns a short reason string, or None if fine.
# ---------------------------------------------------------------------------

def check_too_short(word_count):
    """Return "too short (N words)" if word_count < MIN_WORDS, else None."""

    if word_count < MIN_WORDS:
        return f"Too short ({word_count} words)"
    return None

def check_too_long(word_count, median_words, form="10-Q"):
    """Return "too long (Nx median)" if word_count is over the form's multiple of median_words, else None.

    median_words is the ticker's 10-Q median (add_median_words()). 10-Ks are compared
    against it too, with their own, higher multiple: a 10-K-only median would be the
    median of just 2 filings, which neither can exceed by 2x, so the check could never fire.
    """
    max_multiple = MAX_MULTIPLE_10K if form == "10-K" else MAX_MULTIPLE_10Q
    ratio = word_count / median_words

    if ratio > max_multiple:
        return f"Too long ({ratio:.1f}x median)"
    return None

def check_heading(text):
    """Return "no MD&A heading" if the first HEADING_CHARS characters don't look like MD&A, else None.

    Plan suggests looking for "Management's Discussion", "MD&A" or "Overview".
    Things to handle: capitalisation (MANAGEMENT'S), curly vs straight apostrophes
    (’ vs '), and whatever else Step 1 shows the real files start with.
    """

    header_list = ["Management's Discussion", "MANAGEMENT’S DISCUSSION", "MD&A", "Overview"]

    for header in header_list:
        if header.lower() in text[:HEADING_CHARS].lower():
            return None
    return "No MD&A header found"


def add_median_words(manifest: pd.DataFrame):
    """Return a Series (same index as manifest) with the ticker's 10-Q median word_count on every row.

    10-Q median only, because 10-K MD&A is up to ~4x longer (XOM) and would pull the
    median up; 10-K rows get the same 10-Q median and are judged with their own multiple
    in check_too_long().
    """
    # where() turns 10-K rows into NaN; median skips NaN; transform copies each ticker's
    # median back onto all its rows, 10-Ks included
    q_words = manifest["word_count"].where(manifest["form"] == "10-Q")
    return q_words.groupby(manifest["ticker"]).transform("median")


# ---------------------------------------------------------------------------
# Plumbing: reading files, combining flags, writing the manifest.
# ---------------------------------------------------------------------------

def read_mdna(file_path):
    """MD&A text for a manifest row, or None if ingest saved no file for it."""
    if pd.isna(file_path):
        return None
    return (ROOT / file_path).read_text(encoding="utf-8")


def count_words(text):
    # Whitespace split: crude (counts "$1.2" and "--" as words) but consistent across
    # files, which is all a relative length check needs.
    return len(text.split()) if text else 0


def validate(manifest):
    """Return a copy of the manifest with VALIDATION_COLUMNS filled in, plus the texts.

    Texts are returned (not stored in the manifest) so the report can show first lines.
    """
    # Drop columns from an earlier validate run, so reruns don't stack duplicates
    manifest = manifest.drop(columns=VALIDATION_COLUMNS, errors="ignore").copy()

    texts = manifest["file_path"].map(read_mdna)
    manifest["word_count"] = texts.map(count_words)
    manifest["median_words"] = add_median_words(manifest)

    reasons = []
    for (_, row), text in zip(manifest.iterrows(), texts):
        if text is None:
            # No file at all: nothing else to check
            reasons.append("no MD&A file")
            continue
        found = [
            check_too_short(row["word_count"]),
            check_too_long(row["word_count"], row["median_words"], row["form"]),
            check_heading(text),
        ]
        # A file can fail more than one check; keep all reasons
        reasons.append("; ".join(r for r in found if r) or None)

    manifest["flag_reason"] = reasons
    manifest["review_note"] = manifest["accession_no"].map(REVIEWED_OK)
    # Only a flagged filing can be reviewed OK; a note on an unflagged one changes nothing
    manifest["valid"] = manifest["flag_reason"].isna() | manifest["review_note"].notna()
    return manifest, texts


def print_report(manifest, texts, n_chars=300):
    """Word counts per filing, then every flagged file with its opening text, for eyeballing."""
    cols = ["ticker", "form", "period_of_report", "word_count", "median_words", "valid"]
    print(manifest[cols].to_string(index=False))

    flagged = manifest[manifest["flag_reason"].notna()]
    reviewed = flagged["review_note"].notna()
    print(f"\n{len(flagged)} of {len(manifest)} filings flagged "
          f"({reviewed.sum()} reviewed OK, {(~reviewed).sum()} to check)")
    for i, row in flagged.iterrows():
        print(f"\n--- {row['ticker']} {row['form']} {row['period_of_report']}: {row['flag_reason']}")
        if pd.notna(row["review_note"]):
            # Already eyeballed: show the note instead of the text
            print(f"Reviewed OK: {row['review_note']}")
        elif texts[i]:
            print(texts[i][:n_chars].strip())


def selftest():
    """Toy inputs for the four TODO functions. Expectations use the plan's default thresholds."""
    assert check_too_short(100) is not None
    assert check_too_short(5_000) is None

    assert check_too_long(40_000, 10_000) is not None
    assert check_too_long(12_000, 10_000) is None
    assert check_too_long(40_000, 10_000, "10-K") is None       # 4x is normal-ish for a 10-K
    assert check_too_long(60_000, 10_000, "10-K") is not None

    assert check_heading("Item 2. Management's Discussion and Analysis of Financial Condition") is None
    assert check_heading("ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS") is None   # curly apostrophe
    assert check_heading("Overview\nWe are a technology company...") is None
    assert check_heading("Issuer Purchases of Equity Securities. The following table...") is not None
    assert check_heading("x" * HEADING_CHARS + " Management's Discussion") is not None  # too late

    toy = pd.DataFrame({"ticker": ["A", "A", "A", "B"], "form": ["10-Q", "10-Q", "10-K", "10-Q"],
                        "word_count": [100, 200, 900, 50]})
    medians = add_median_words(toy)
    assert list(medians.index) == list(toy.index)
    # Ticker A's 10-Q median (100, 200 -> 150) on all A rows, including its 10-K
    assert medians.tolist() == [150.0, 150.0, 150.0, 50.0], medians.tolist()
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="dry run on these tickers: print, don't write")
    parser.add_argument("--selftest", action="store_true", help="run toy checks and exit")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    # fiscal_year_end as str, or "0630" comes back as 630 and is written that way
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"fiscal_year_end": str})
    if args.tickers:
        manifest = manifest[manifest["ticker"].isin(args.tickers)].reset_index(drop=True)

    manifest, texts = validate(manifest)
    print_report(manifest, texts)

    if args.tickers:
        print("\nDry run (--tickers): manifest not written")
    else:
        manifest.to_csv(MANIFEST_PATH, index=False)
        print(f"\nWrote {len(manifest)} rows to {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
