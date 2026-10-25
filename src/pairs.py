"""Decide which filings get compared with which, from data/processed/manifest.csv.

Pairing rules:
  yoy     10-Q vs the 10-Q for the same fiscal quarter a year earlier. Main comparison:
          removes seasonal wording (e.g. holiday quarter language at WMT).
  qoq     10-Q vs the previous 10-Q in the same fiscal year (Q2 vs Q1, Q3 vs Q2).
          Q1 has no qoq pair: its previous quarter is a 10-K, structured too differently.
  annual  10-K vs the prior year's 10-K.

Filings are matched on fiscal year and quarter, not calendar dates, so companies with
different fiscal years (MSFT June, NVDA/WMT January) line up correctly.
Amendments are left out; pairs use the original filings.

Run from the project root:
    python -m src.pairs
"""

import pandas as pd

from src.config import ROOT, DATA_DIR

MANIFEST_PATH = DATA_DIR / "manifest.csv"
PAIRS_PATH = DATA_DIR / "pairs.csv"

KEEP = ["accession_no", "form", "period_of_report", "fiscal_year", "fiscal_quarter",
        "acceptance_datetime", "file_path"]


def build_pairs(manifest):
    """Return one row per comparison: ticker, pair_type, then old_* and new_* filing columns."""
    filings = manifest[~manifest["is_amendment"] & manifest["file_path"].notna()]
    filings = filings[["ticker"] + KEEP]
    old = filings.rename(columns={c: f"old_{c}" for c in KEEP})
    new = filings.rename(columns={c: f"new_{c}" for c in KEEP})

    # Every combination of two filings for the same ticker, then keep the ones a rule allows
    both = new.merge(old, on="ticker")
    same_q = both["new_fiscal_quarter"] == both["old_fiscal_quarter"]
    prior_year = both["new_fiscal_year"] == both["old_fiscal_year"] + 1
    same_year = both["new_fiscal_year"] == both["old_fiscal_year"]
    prev_q = both["new_fiscal_quarter"] == both["old_fiscal_quarter"] + 1
    both_10q = (both["new_form"] == "10-Q") & (both["old_form"] == "10-Q")
    both_10k = (both["new_form"] == "10-K") & (both["old_form"] == "10-K")

    both.loc[both_10q & same_q & prior_year, "pair_type"] = "yoy"
    both.loc[both_10q & same_year & prev_q, "pair_type"] = "qoq"
    both.loc[both_10k & prior_year, "pair_type"] = "annual"

    pairs = both.dropna(subset=["pair_type"])
    cols = ["ticker", "pair_type"] + [f"old_{c}" for c in KEEP] + [f"new_{c}" for c in KEEP]
    return pairs[cols].sort_values(["ticker", "pair_type", "new_period_of_report"]).reset_index(drop=True)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    pairs = build_pairs(manifest)
    pairs.to_csv(PAIRS_PATH, index=False)
    print(f"Wrote {len(pairs)} pairs to {PAIRS_PATH.relative_to(ROOT)}\n")
    print(pairs.groupby(["ticker", "pair_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
