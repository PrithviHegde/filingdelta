"""Clean extracted MD&A text before diffing.

Steps so far:
  trim_item8   cut text that ran past the end of MD&A into Item 8 (financial statements)

Still to do (docs/plan.md, Day 2 task 3): paragraphs, page numbers and repeated headers,
re-joining page-split paragraphs, section labels, table tags.

Raw MD&A files are never modified: cleaned text goes to {period}_mdna_clean.txt next to
them, so every cleaning rule can be checked against (and undone from) the original.

Run from the project root:
    python -m src.clean                  # clean all filings in the manifest
    python -m src.clean --tickers MSFT   # some tickers only
"""

import argparse
import re

import pandas as pd

from src.config import ROOT, DATA_DIR

MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Headings that open Item 8 in the 10-Ks where edgartools' Item 7 runs on (CVX, JPM, XOM,
# found by validation on 2026-10-24). Matched only as a whole line, so a table of contents
# entry ("Management's Report on Internal Control ...  73") doesn't trigger a cut:
# XOM's 2024 10-K lists these headings on line 15, and a plain text search would have cut
# almost the whole MD&A.
ITEM8_HEADINGS = re.compile(
    r"^\s*("
    r"management's responsibility for financial statements"          # CVX opens Item 8 with this
    r"|management's report on internal control over financial reporting"
    r"|report of independent registered public accounting firm"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# A cut that removes more than this share of the text is suspicious (heading matched
# somewhere unexpected): it is still applied, but printed as a warning to check by eye.
# The largest real overrun found was 20% (CVX 2025 10-K).
WARN_TRIM_SHARE = 0.3


def normalise_apostrophes(text):
    # Curly ’ to straight ', so one pattern matches both. Same length, so positions
    # found in the normalised text are valid in the original.
    return text.replace("’", "'")


def trim_item8(text):
    """Return (text cut at the first Item 8 heading, number of characters removed).

    Text without such a heading is returned unchanged with 0 removed.
    """
    match = ITEM8_HEADINGS.search(normalise_apostrophes(text))
    if match is None:
        return text, 0
    return text[:match.start()].rstrip() + "\n", len(text) - match.start()


def clean_path(file_path):
    """data/processed/MSFT/2025-09-30_mdna.txt -> data/processed/MSFT/2025-09-30_mdna_clean.txt"""
    path = ROOT / file_path
    return path.with_name(f"{path.stem}_clean{path.suffix}")


def clean_filing(file_path):
    """Clean one MD&A file, write the result, and return a summary row for the report."""
    raw = (ROOT / file_path).read_text(encoding="utf-8")
    text, removed_chars = trim_item8(raw)

    out_path = clean_path(file_path)
    out_path.write_text(text, encoding="utf-8")

    raw_words, removed_words = len(raw.split()), len(raw.split()) - len(text.split())
    return {
        "file": file_path,
        "raw_words": raw_words,
        "item8_words_removed": removed_words,
        "share_removed": round(removed_words / raw_words, 3) if raw_words else 0,
        # Last line kept, so the report shows where MD&A now ends
        "ends_with": text.strip().splitlines()[-1][:80] if removed_chars else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="default: all tickers in the manifest")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"fiscal_year_end": str})
    manifest = manifest[manifest["file_path"].notna()]
    if args.tickers:
        manifest = manifest[manifest["ticker"].isin(args.tickers)]

    report = pd.DataFrame([clean_filing(f) for f in manifest["file_path"]])
    trimmed = report[report["item8_words_removed"] > 0]

    print(f"Cleaned {len(report)} files; Item 8 trimmed from {len(trimmed)}")
    if len(trimmed):
        print(trimmed.to_string(index=False))
    for _, row in trimmed[trimmed["share_removed"] > WARN_TRIM_SHARE].iterrows():
        print(f"  WARNING: {row['file']} lost {row['share_removed']:.0%} of its words, check the cut")


if __name__ == "__main__":
    main()
