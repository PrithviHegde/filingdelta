"""Clean extracted MD&A text and split it into labelled paragraphs, ready for diffing.

The steps, in the order they run on each filing:
  1. trim_item8         cut text that ran past the end of MD&A into Item 8 (financial statements)
  2. split_blocks       split the text into blocks at blank lines (one block ~ one paragraph)
  3. drop_junk          remove page numbers and running page headers / footers
  4. join_page_splits   glue back paragraphs that a page break cut in two
  5. block_kind         label each block "heading", "table" or "text"
  6. section            give every block the most recent heading above it

Outputs:
  data/processed/{ticker}/{period}_mdna_clean.txt   cleaned text, one block per paragraph
                                                    (for reading by eye)
  data/processed/paragraphs.csv                     one row per block, all filings
                                                    (what diff.py reads)

Raw MD&A files are never modified, so every rule can be checked against the original.

Run from the project root:
    python -m src.clean                  # clean all filings in the manifest
    python -m src.clean --tickers MSFT   # some tickers only (paragraphs.csv not written)
"""

import argparse
import re
from collections import Counter

import pandas as pd

from src.config import ROOT, DATA_DIR

MANIFEST_PATH = DATA_DIR / "manifest.csv"
PARAGRAPHS_PATH = DATA_DIR / "paragraphs.csv"

# ---------------------------------------------------------------------------
# Step 1: trim Item 8
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 2: split into blocks
# ---------------------------------------------------------------------------

def split_blocks(text):
    """Split text at blank lines and return the non-empty blocks, stripped.

    edgartools puts a blank line between paragraphs, so a block is usually one paragraph,
    one heading, one table, or one page number. Lines inside a block (e.g. table rows)
    stay together.
    """
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


# ---------------------------------------------------------------------------
# Step 3: drop page numbers and running headers / footers
# ---------------------------------------------------------------------------

# Page junk comes in a cluster at every page break:
#   page number:     "31", "Page 31", "35." (GOOGL)
#   footer:          one line with the page number at its start or end, set off by 2+ spaces:
#                    "JPMorgan Chase & Co./2025 Form 10-K  160", "34  ConocoPhillips 2026 Q2 10-Q"
#   running header:  a short line repeated on many pages, right next to the page number:
#                    "PART I" / "Item 2" (MSFT), "Table of Contents  Alphabet Inc." (GOOGL)
#
# Why "next to a page break" matters: the first version dropped every short line that
# repeated 3+ times, and that also threw away real headings that repeat per segment
# ("More Personal Computing", "Google Cloud", "Net Revenues") and real sentences that
# differ only in their numbers ("Revenue increased $5.1 billion or 17%."). Real junk is
# almost always within a block or two of a page break; real headings mostly aren't.
#
# Digits are masked before comparing, so "... Form 10-K  160" and "... Form 10-K  161" count
# as the same footer.
REPEAT_MAX_WORDS = 20     # CVX's header is 15 words: "Management’s Discussion and Analysis of ... Financial Table of Contents"
REPEAT_MIN_COUNT = 3
PAGE_BREAK_WINDOW = 2     # "near a page break" = within this many blocks of one
NEAR_BREAK_SHARE = 0.5    # a header must be near a break in at least half its appearances


def is_page_number(block):
    return re.fullmatch(r"(page\s*)?\d{1,3}\.?", block, re.IGNORECASE) is not None


def has_page_number_at_edge(block):
    # Footer shape: page number at the start or end of a single line, after 2+ spaces
    return "\n" not in block and re.search(r"^\d{1,3}\s{2,}\S|\S\s{2,}\d{1,3}$", block) is not None


def mask_digits(block):
    # "JPMorgan Chase & Co./2025 Form 10-K  160" -> "JPMorgan Chase & Co./#### Form ##-K  ###"
    return re.sub(r"\d", "#", block)


def could_be_page_furniture(block):
    """Short single line that doesn't end like a sentence. Only these can be headers/footers.

    The sentence-end rule keeps "Net income was $2.4 billion, up 41%." even if it happens
    to sit next to page breaks. "Alphabet Inc." is the one exception seen: it ends with a
    full stop but is part of GOOGL's page header.
    """
    ends_like_sentence = block.endswith(SENTENCE_END) and not block.endswith("Inc.")
    return "\n" not in block and len(block.split()) <= REPEAT_MAX_WORDS and not ends_like_sentence


def drop_junk(blocks):
    """Remove page numbers, footers and running headers. Returns (kept blocks, dropped blocks)."""
    masked = [mask_digits(b) for b in blocks]

    # 1. Which short lines repeat 3+ times in this filing?
    counts = Counter(m for b, m in zip(blocks, masked) if could_be_page_furniture(b))
    repeats = [could_be_page_furniture(b) and counts[m] >= REPEAT_MIN_COUNT
               for b, m in zip(blocks, masked)]

    # 2. Page breaks: page numbers, and repeated lines shaped like a footer
    is_break = [is_page_number(b) or (rep and has_page_number_at_edge(b))
                for b, rep in zip(blocks, repeats)]

    # 3. For each block, is there a page break within PAGE_BREAK_WINDOW blocks of it?
    near_break = [any(is_break[max(0, i - PAGE_BREAK_WINDOW): i + PAGE_BREAK_WINDOW + 1])
                  for i in range(len(blocks))]

    # 4. A running header repeats AND sits near a break in at least half its appearances
    near_counts = Counter(m for m, rep, near in zip(masked, repeats, near_break) if rep and near)
    headers = {m for m, n in counts.items()
               if n >= REPEAT_MIN_COUNT and near_counts[m] / n >= NEAR_BREAK_SHARE}

    kept, dropped = [], []
    for block, m, brk in zip(blocks, masked, is_break):
        if brk or m in headers:
            dropped.append(block)
        else:
            kept.append(block)
    return kept, dropped


# ---------------------------------------------------------------------------
# Step 4: re-join paragraphs split by a page break
# ---------------------------------------------------------------------------

# Once page numbers and headers are gone, a paragraph cut by a page break shows up as two
# text blocks in a row, e.g. (KO)
#   "...represents the amount of concentrates, syrups,"  +  "source waters and powders/minerals..."
#
# Two signs that the second block carries on the first one's sentence:
#   a. it starts with a lowercase letter: a new paragraph, bullet ("•..."), footnote
#      ("(a)...", "*...") or heading never does; or
#   b. the first block's last word is lowercase ("the", "and", "integrated") or it ends in a
#      comma ("...in integrated" + "Upstream Permian Basin ..."), and the second starts with
#      a letter (so not a bullet, footnote or number).
# (b) is needed for XOM's 2026 10-Qs, where edgartools put a blank line after every printed
# line, so each paragraph arrives as a dozen line-sized blocks, many starting with a capital.
# It checks the whole last word, not just the last letter: blocks ending in a capitalised
# word ("...Financial Table of Contents" page header at the end of a CVX block, a chart
# caption "Source: ExxonMobil 2024 Global Outlook") are usually not mid-sentence.
# An earlier version that joined any two text blocks glued tables to the paragraph after
# them and bullets to each other.
# Cost: some real splits stay as two blocks: second half starting with "(" (KO "...partners"
# + "(collectively, ..."), or first half ending in a capitalised word ("...in Greater" + "China").
SENTENCE_END = (".", ":", ";", "?", "!", "”", '"', ")")
JOIN_MIN_WORDS = 15


def is_page_split(previous, block):
    """True if `block` looks like the rest of `previous`, cut off by a page break."""
    if not (
        block_kind(previous) == "text"
        and len(previous.split()) >= JOIN_MIN_WORDS     # short blocks without a full stop are headings
        and not previous.endswith(SENTENCE_END)          # previous stopped mid-sentence
        and block_kind(block) == "text"
    ):
        return False
    starts_lowercase = block[0].islower()                                           # sign (a)
    last_word = previous.split()[-1]
    previous_ends_mid_phrase = last_word[0].islower() or previous.endswith(",")     # sign (b)
    return starts_lowercase or (previous_ends_mid_phrase and block[0].isalpha())


def join_page_splits(blocks):
    """Glue each block to the one before it when is_page_split() says so. Returns (blocks, n_joins)."""
    joined = []
    n_joins = 0
    for block in blocks:
        if joined and is_page_split(joined[-1], block):
            joined[-1] = joined[-1] + " " + block
            n_joins += 1
        else:
            joined.append(block)
    return joined, n_joins


# ---------------------------------------------------------------------------
# Step 5: label each block
# ---------------------------------------------------------------------------

# Tables are tagged so diff.py can skip them: their changes are number updates, not language.
# A block counts as a table if either
#   - at least 40% of its words contain a digit: "Worldwide  5%  4%  4%  6%" (KO), or
#   - at least half its lines look like table rows, i.e. a cell gap (2+ spaces) followed by
#     a number: "Sales and other operating revenues  $  48,926  $51,922" (CVX). This catches
#     tables whose long row labels keep the digit share under 40%.
TABLE_MIN_NUMBER_SHARE = 0.4
TABLE_MIN_ROW_SHARE = 0.5
TABLE_ROW = re.compile(r"\s{2,}[$(]*\s*\d")
HEADING_MAX_WORDS = 12


def number_share(block):
    # Share of whitespace-separated tokens that contain a digit
    tokens = block.split()
    return sum(any(ch.isdigit() for ch in t) for t in tokens) / len(tokens)


def row_share(block):
    # Share of lines that look like a table row (cell gap followed by a number)
    lines = block.splitlines()
    return sum(TABLE_ROW.search(line) is not None for line in lines) / len(lines)


def block_kind(block):
    """Return "table", "heading" or "text" for one block.

    heading: short, one line, starts with a capital letter, no sentence punctuation and no
             colon ("OVERVIEW", "Seasonality", "Industry Trends and Opportunities").
             The colon rule keeps lines like "Source: ExxonMobil 2025 Global Outlook" out.
    table:   mostly numbers, or mostly table rows (see the constants above).
    text:    everything else, including bullets ("•Revenue increased 12%.") and
             footnotes ("(a)As of June 30, 2026, ...").
    """
    if number_share(block) >= TABLE_MIN_NUMBER_SHARE or row_share(block) >= TABLE_MIN_ROW_SHARE:
        return "table"
    is_heading = (
        "\n" not in block
        and len(block.split()) <= HEADING_MAX_WORDS
        and block[0].isupper()
        and not block.endswith(SENTENCE_END)
        and ":" not in block
    )
    return "heading" if is_heading else "text"


# ---------------------------------------------------------------------------
# Whole filing
# ---------------------------------------------------------------------------

def clean_path(file_path):
    """data/processed/MSFT/2025-09-30_mdna.txt -> data/processed/MSFT/2025-09-30_mdna_clean.txt"""
    path = ROOT / file_path
    return path.with_name(f"{path.stem}_clean{path.suffix}")


def clean_filing(filing):
    """Run all steps on one manifest row. Returns (paragraph rows, summary dict, dropped blocks)."""
    raw = (ROOT / filing["file_path"]).read_text(encoding="utf-8")

    text, _ = trim_item8(raw)
    blocks = split_blocks(text)
    blocks, dropped = drop_junk(blocks)
    blocks, n_joins = join_page_splits(blocks)

    # Label every block and carry the latest heading down as its section
    rows = []
    section = None
    for i, block in enumerate(blocks):
        kind = block_kind(block)
        if kind == "heading":
            section = block
        rows.append({
            "accession_no": filing["accession_no"],
            "ticker": filing["ticker"],
            "form": filing["form"],
            "period_of_report": filing["period_of_report"],
            "para_no": i,
            "kind": kind,
            "section": section,
            "words": len(block.split()),
            "text": block,
        })

    # Cleaned text for reading by eye: blocks separated by a blank line, as in the raw file
    clean_path(filing["file_path"]).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    raw_words = len(raw.split())
    item8_words = raw_words - len(text.split())
    kinds = Counter(row["kind"] for row in rows)
    summary = {
        "ticker": filing["ticker"],
        "form": filing["form"],
        "period": filing["period_of_report"],
        "raw_words": raw_words,
        "item8_removed": item8_words,
        "item8_share": round(item8_words / raw_words, 3),
        "junk_dropped": len(dropped),
        "joins": n_joins,
        "text": kinds["text"],
        "heading": kinds["heading"],
        "table": kinds["table"],
    }
    return rows, summary, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="only these tickers; paragraphs.csv is not written")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"fiscal_year_end": str})
    manifest = manifest[manifest["file_path"].notna()]
    if args.tickers:
        manifest = manifest[manifest["ticker"].isin(args.tickers)]

    all_rows, summaries, dropped_by_ticker = [], [], {}
    for _, filing in manifest.iterrows():
        rows, summary, dropped = clean_filing(filing)
        all_rows.extend(rows)
        summaries.append(summary)
        dropped_by_ticker.setdefault(filing["ticker"], Counter()).update(mask_digits(b) for b in dropped)

    report = pd.DataFrame(summaries)
    print(report.to_string(index=False))

    for _, row in report[report["item8_share"] > WARN_TRIM_SHARE].iterrows():
        print(f"  WARNING: {row['ticker']} {row['period']} lost {row['item8_share']:.0%} to the Item 8 trim, check the cut")

    # What was thrown away as junk, so it can be checked by eye (digits shown as #)
    print("\nMost common dropped blocks per ticker:")
    for ticker, counts in dropped_by_ticker.items():
        top = ", ".join(f"{n}x {block[:50]!r}" for block, n in counts.most_common(4))
        print(f"  {ticker}: {top}")

    if args.tickers:
        print("\n--tickers given: paragraphs.csv not written")
    else:
        pd.DataFrame(all_rows).to_csv(PARAGRAPHS_PATH, index=False)
        print(f"\nWrote {len(all_rows)} paragraphs to {PARAGRAPHS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
