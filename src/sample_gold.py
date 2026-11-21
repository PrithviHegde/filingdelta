"""Pick the gold-set items for the change classifier, reproducibly.

Population: rows of data/processed/changes_embedding.csv whose change is reworded, added or
removed (the rows the classifier labels). Unchanged and numbers_only rows are never labelled.

Design:
- Per ticker, a fixed quota: 2 reworded, 1 added, 1 removed (44 items), then 6 more reworded
  drawn from all tickers (50 in total). A plain random sample would be a third JPM (JPM has
  3,113 of the 9,687 rows) and would leave small tickers with one item or none; the quota
  keeps every company and change type in the set. The resulting mix (56% reworded, 22% added,
  22% removed) stays close to the population's (60 / 22 / 18).
- Excluded: the 5 smoke-test items and the MS item used while writing the prompt (both seen
  while tuning it, so they would leak into the evaluation), and the rows already in
  labels/diff_handcheck.csv (labelled for a different question, and seen before).
- Split before labelling: 15 test rows (scored once, at the end) and 35 dev rows (for
  comparing models and prompts), drawn at random within the sample.
- Fixed seed: rerunning gives the same 50 rows, as long as changes_embedding.csv is unchanged.

Writes labels/gold_items.csv: one row per item with a stable item_id, the split, and
everything an annotator needs (ticker, pair type, periods, sections, change type, OLD, NEW).
The texts are copied in, so the gold set doesn't depend on regenerating data/.

Run from the project root:
    python -m src.sample_gold            # write labels/gold_items.csv
    python -m src.sample_gold --dry-run  # print the sample, don't write
"""

import argparse

import numpy as np
import pandas as pd

from src.config import DATA_DIR, ROOT

CHANGES_PATH = DATA_DIR / "changes_embedding.csv"
HANDCHECK_PATH = ROOT / "labels" / "diff_handcheck.csv"
GOLD_ITEMS_PATH = ROOT / "labels" / "gold_items.csv"

LABELLED_CHANGES = ["reworded", "added", "removed"]
QUOTA_PER_TICKER = {"reworded": 2, "added": 1, "removed": 1}
EXTRA_REWORDED = 6
N_TEST = 15
SEED = 20261120

# Rows seen while writing the prompt (notebook 05): the smoke test's picks are reproduced
# here with its own seed and filters, and the MS item is hand-check row 13.
SMOKE_SEED = 0

KEY = ["ticker", "pair_type", "old_period", "new_period"]


def item_id(r):
    """Stable id: the two filings plus the two paragraph numbers (one side may be empty)."""
    para = lambda x: "" if pd.isna(x) else str(int(x))
    return f"{r.old_accession_no}|{r.new_accession_no}|{para(r.old_para_no)}|{para(r.new_para_no)}"


def smoke_test_rows(changes):
    """The five rows the notebook's smoke test picked (same seed, same filters, same order)."""
    rng = np.random.default_rng(SMOKE_SEED)
    def pick(mask):
        rows = changes[mask]
        return rows.index[rng.integers(len(rows))]
    reworded = changes.change == "reworded"
    new_text = changes.new_text.fillna("")
    return [
        pick(changes.change == "added"),
        pick(changes.change == "removed"),
        pick(reworded & (changes.score >= 0.98)),
        pick((changes.ticker == "JPM") & reworded & new_text.str.contains("allowance for credit losses")),
        pick(reworded & new_text.str.contains(r"\bexpect", case=False)),
    ]


def handcheck_rows(changes):
    """Rows of changes_embedding.csv holding a paragraph from labels/diff_handcheck.csv."""
    hc = pd.read_csv(HANDCHECK_PATH)
    found = set()
    for _, r in hc.iterrows():
        same_pair = (changes[KEY] == r[KEY]).all(axis=1)
        for side in ["old_text", "new_text"]:
            if pd.notna(r[side]):
                found |= set(changes.index[same_pair & (changes[side] == r[side])])
    return found


def sample(changes):
    rng = np.random.default_rng(SEED)
    pool = changes[changes.change.isin(LABELLED_CHANGES)]
    excluded = set(smoke_test_rows(changes)) | handcheck_rows(changes)
    pool = pool.drop(index=[i for i in excluded if i in pool.index])

    picked = []
    for ticker in sorted(pool.ticker.unique()):
        for change, n in QUOTA_PER_TICKER.items():
            rows = pool[(pool.ticker == ticker) & (pool.change == change)]
            picked += list(rng.choice(rows.index, size=n, replace=False))
    rest = pool[(pool.change == "reworded") & ~pool.index.isin(picked)]
    picked += list(rng.choice(rest.index, size=EXTRA_REWORDED, replace=False))

    gold = pool.loc[picked].copy()
    gold["item_id"] = [item_id(r) for r in gold.itertuples()]
    # Test rows drawn within each change type (~30% of each), so the 15 test rows have the same
    # mix as the dev rows; a plain random draw of 15 gave 2 added vs 5 removed.
    test = set()
    for change, rows in gold.groupby("change"):
        n = round(len(rows) * N_TEST / len(gold))
        test |= set(rng.choice(rows.index, size=n, replace=False))
    gold["split"] = ["test" if i in test else "dev" for i in gold.index]
    # Shuffle so the labelling order isn't grouped by ticker (fatigue and anchoring on one company)
    gold = gold.sample(frac=1, random_state=SEED).reset_index(drop=True)
    cols = ["item_id", "split", "ticker", "pair_type", "old_period", "new_period",
            "old_section", "new_section", "change", "old_text", "new_text"]
    return gold[cols], len(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the sample, don't write")
    args = parser.parse_args()

    changes = pd.read_csv(CHANGES_PATH)
    gold, n_excluded = sample(changes)

    assert gold.item_id.is_unique, "duplicate items in the sample"
    print(f"{len(gold)} items ({n_excluded} rows excluded as already seen)")
    print(gold.groupby(["split", "change"]).size().unstack(fill_value=0).to_string())
    print("\nper ticker:", gold.ticker.value_counts().sort_index().to_dict())
    print("pair types:", gold.pair_type.value_counts().to_dict())

    if args.dry_run:
        print("\n--dry-run: not written")
    else:
        gold.to_csv(GOLD_ITEMS_PATH, index=False)
        print(f"\nWrote {GOLD_ITEMS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
