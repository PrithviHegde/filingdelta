"""Score a classifier run against the adjudicated gold labels (labels/gold_labels.csv).

Reads the results file that src/classify.py wrote for the same model, prompt version and
split, and reports per field: exact agreement and Cohen's kappa (agreement corrected for
chance; the gold set's materiality is about half 0s, so raw accuracy flatters a model that
over-predicts 0), the confusion matrix, per-class recall, and for materiality, the share
within one level (it is ordinal: 0 vs 2 is a worse miss than 0 vs 1). The bad_match gold item
is left out: it has no labels.

The human reference is the inter-annotator agreement measured while building the gold set
(kappa: topic 0.79, materiality 0.41, direction 0.48, forward_looking 1.00): a model near
those values is as consistent with the gold set as one careful annotator is with another.

Run from the project root:
    python -m src.evaluate --split dev                       # qwen3.5:9b, current prompt version
    python -m src.evaluate --split dev --model gemma4:12b
    python -m src.evaluate --split dev --errors              # also list every disagreement
"""

import argparse

import numpy as np
import pandas as pd

from src.classify import MODEL, PROMPT_VERSION, read_results, results_path
from src.config import ROOT

GOLD_LABELS_PATH = ROOT / "labels" / "gold_labels.csv"
GOLD_ITEMS_PATH = ROOT / "labels" / "gold_items.csv"
FIELDS = ["topic", "materiality", "direction", "forward_looking"]
HUMAN_KAPPA = {"topic": 0.79, "materiality": 0.41, "direction": 0.48, "forward_looking": 1.00}


def cohen_kappa(a, b):
    """(observed agreement, Cohen's kappa) for two equal-length label lists."""
    a, b = list(a), list(b)
    n = len(a)
    observed = np.mean([x == y for x, y in zip(a, b)])
    expected = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return observed, kappa


def load(split, model):
    """Gold rows of the split joined to the model's records; labels as lowercase strings."""
    gold = pd.read_csv(GOLD_LABELS_PATH, dtype=str, keep_default_na=False)
    gold = gold[(gold.split == split) & (gold.flag != "bad_match")]
    path = results_path(model, f"gold_{split}")
    records = pd.DataFrame(read_results(path).values())
    if records.empty:
        raise SystemExit(f"no results in {path.relative_to(ROOT)}: run src.classify first")
    ok = records[records.status == "ok"]
    joined = gold.merge(ok, on="item_id", how="left", suffixes=("_gold", "_model"))
    for f in FIELDS:
        joined[f + "_model"] = joined[f + "_model"].map(lambda v: "" if pd.isna(v) else str(v).lower())
    return joined, records


def report(d, records, model, split):
    missing = (d.topic_model == "").sum()
    print(f"{model}, prompt {PROMPT_VERSION}, {split} split: {len(d)} scored gold items "
          f"({missing} without a successful result, counted as wrong)")
    print(f"run: {(records.status == 'ok').sum()} ok, {(records.status != 'ok').sum()} failed, "
          f"{records.seconds.mean():.1f} s and {records.output_tokens.mean():.0f} output tokens per item\n")

    print(f"{'field':16s} {'agreement':>9s} {'kappa':>6s} {'human kappa':>11s}")
    for f in FIELDS:
        obs, k = cohen_kappa(d[f + "_gold"], d[f + "_model"])
        print(f"{f:16s} {obs:9.0%} {k:6.2f} {HUMAN_KAPPA[f]:11.2f}")

    scored = d[d.materiality_model != ""]
    g, m = scored.materiality_gold.astype(int), scored.materiality_model.astype(int)
    print(f"\nmateriality within one level: {np.mean(abs(g - m) <= 1):.0%}")
    for f in ["materiality", "direction"]:
        print(f"\n{f}: rows = gold, columns = model")
        print(pd.crosstab(d[f + "_gold"], d[f + "_model"]).to_string())
        recall = {c: f"{(grp[f + '_model'] == c).mean():.0%} of {len(grp)}"
                  for c, grp in d.groupby(f + "_gold")}
        print("recall:", recall)

    # Rows where the model breaks the guidelines' consistency rules (0 <-> not_applicable)
    viol = scored[(scored.materiality_model == "0") != (scored.direction_model == "not_applicable")]
    print(f"\nconsistency-rule violations: {len(viol)}")


def list_errors(d):
    items = pd.read_csv(GOLD_ITEMS_PATH)
    number = {iid: i + 1 for i, iid in enumerate(items.item_id)}
    print("\nDisagreements (# = row in the labelling sheet):")
    for _, r in d.iterrows():
        wrong = [f for f in FIELDS if r[f + "_gold"] != r[f + "_model"]]
        if not wrong:
            continue
        item = items.loc[number[r.item_id] - 1]
        diffs = ", ".join(f"{f} {r[f + '_gold']}->{r[f + '_model']}" for f in wrong)
        flag = " [superseded]" if "superseded" in r.note else ""
        print(f"#{number[r.item_id]:2d} {item.ticker:5s} {item.change:8s} {diffs}{flag}")
        print(f"     model reason: {r.reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--errors", action="store_true", help="list every disagreement")
    args = parser.parse_args()

    d, records = load(args.split, args.model)
    report(d, records, args.model, args.split)
    if args.errors:
        list_errors(d)


if __name__ == "__main__":
    main()
