"""Classify each MD&A change with a local LLM: topic, materiality, direction, forward_looking.

Input: rows of data/processed/changes_embedding.csv with change reworded / added / removed
(or the gold items in labels/gold_items.csv, which have the same columns).
Model: a local model through Ollama, with structured output (the JSON schema of ChangeLabel)
and fixed decoding settings. Rules: the same as labels/guidelines.md.

Results: one JSONL file per model, prompt version and item set in data/processed/classified/,
one line per item, appended as it finishes. Rerunning the same command skips items already
classified and retries failed ones.

Run from the project root (Ollama must be running):
    python -m src.classify --items gold --split dev        # the 35 dev gold items
    python -m src.classify --items all --tickers MSFT      # one ticker first
    python -m src.classify --items all                     # all 9,687 changes (~12 h)
    python -m src.classify --items gold --split dev --model gemma4:12b
"""


from typing import Literal
import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import time
import ollama
import argparse
from src.sample_gold import item_id as make_item_id
from src.config import DATA_DIR, ROOT

CHANGES_PATH = DATA_DIR / "changes_embedding.csv"
GOLD_ITEMS_PATH = ROOT / "labels" / "gold_items.csv"
LABELLED_CHANGES = ["reworded", "added", "removed"]
OUT_DIR = DATA_DIR / "classified"

MODEL = "qwen3.5:9b"

# v6: direction value neutral_or_mixed renamed no_clear_direction (same meaning). Dev showed
# the models gave a sign to 10 of 12 gold "neutral_or_mixed" items, which were mostly unsigned
# details rather than mixed ones; the old name suggested only the "mixed" case.
PROMPT_VERSION = "v6"

SYSTEM_PROMPT = """\
You compare two versions of a paragraph from the MD&A section of a company's SEC filing
(10-Q or 10-K) and classify how the language changed, from the point of view of an investor.

## Input
Each request gives: ticker, pair type (yoy = same quarter last year, qoq = previous quarter,
annual = previous 10-K), the two periods, the section heading, and the change type:
- reworded: OLD and NEW are versions of the same paragraph
- added: only NEW exists (the paragraph is new in this filing)
- removed: only OLD exists (the paragraph was dropped)

Numbers have already been checked: a change that only updates numbers or dates is never sent.
Every item has a change in wording. Judge what the wording says. Use the numbers as context
for how large a change is, but a number moving is not by itself a reason to call a change material.

Most rewordings are small. Start from materiality 0 and direction not_applicable, and move up
only if you can name what an investor learns from NEW that OLD did not say.

These are always materiality 0:
- the explanation is unchanged and only numbers, dates or periods changed, including an added
  year-to-date (six- or nine-month) comparison next to the quarter
- cross-references ("refer to Note X", "see pages Y"), whatever they point to
- a dropped footnote or caveat that no longer applies
- grammar, punctuation, word order and formatting edits

## Fields
reason: one short sentence naming the key change, quoting the changed words. At most 25 words.

topic: the main subject of the changed content.
Every change gets the topic of its subject, even when trivial; use other only when nothing fits
- revenue_demand: sales, volumes, pricing, customer demand, segment or product revenue
- costs_margins: operating expenses, cost of goods sold, gross margins, SG&A, interest expense, cost-saving initiatives
- outlook_guidance: only explicit company guidance or targets; otherwise pick the subject the statement is about, and mark forward_looking true
- capital_liquidity: debt, cash, buybacks, dividends, capital ratios, credit ratings, funding
- credit_risk: loan losses, allowances, charge-offs, credit quality of loans
- legal_regulatory: litigation, new compliance rules, government investigations, regulatory matters
- macro: inflation, interest rates, foreign exchange, supply chains, broader economic or industry trends, the energy transition
- strategy_mna: mergers, acquisitions, divestitures, strategic shifts, joint ventures, restructuring, capital projects, cost-reduction programmes
- accounting: accounting policies, restatements, critical accounting estimates, revenue recognition
- other: anything else, including business descriptions, cross-references and boilerplate

materiality:
- 0: an investor learns nothing new (see the list above)
- 1: new detail that does not change the picture: a new or changed driver, an extra explanation,
    a clarification, an acquisition the company calls not material
- 2: changes the picture, and only these: a new or removed risk, changed guidance or outlook
    (including changed hedging about the future), a reversed trend (a named revenue or cost
    line flipping direction counts, whatever its size), a new legal or regulatory matter,
    or a major event such as a disruption to operations

direction: is the company's situation or outlook described as better or worse than before?
- more_positive / more_negative / no_clear_direction / not_applicable
- For added paragraphs, judge the new information; for removed ones, the information dropped.
  A removed risk or problem is usually more_positive; an added one more_negative.
- no_clear_direction: the change matters but has no clear sign: a new factual detail, a changed
  driver that is neither better nor worse, or positive and negative elements together.
- not_applicable: only for materiality 0. A change with materiality 1 or 2 always gets a
  direction; use no_clear_direction when it reads neither better nor worse.

forward_looking: true if the paragraph is about the future (expectations, guidance, plans,
risks that may happen); false if it only describes the past or present. Judge the whole
paragraph (NEW, or OLD if removed), not only the words that changed.

## Consistency rules
- materiality 0 -> direction not_applicable.
- cross-references -> topic other.
- materiality 1 or 2 -> direction is not not_applicable.
"""

class ChangeLabel(BaseModel):
    reason: str = Field(description="One sentence, at most 25 words, quoting the changed words")
    topic: Literal["revenue_demand", "costs_margins", "outlook_guidance", "capital_liquidity",
                    "credit_risk", "legal_regulatory", "macro", "strategy_mna", "accounting", "other"]
    materiality: Literal[0, 1, 2]
    direction: Literal["more_positive", "more_negative", "no_clear_direction", "not_applicable"]
    forward_looking: bool
    

def user_message(row):
    """The per-item part of the prompt: context lines, then OLD and/or NEW."""
    lines = [
            f"Ticker: {row.ticker}",
            f"Pair type: {row.pair_type} ({row.old_period} -> {row.new_period})",
            f"Section: {row.new_section if pd.notna(row.new_section) else row.old_section}",
            f"Change type: {row.change}",
        ]
    
    if pd.notna(row.old_text):
        lines.append(f"OLD: {row.old_text}")
    if pd.notna(row.new_text):
        lines.append(f"NEW: {row.new_text}")

    return "\n".join(lines)


# Run the classification on entire dataset

OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 300}
LABEL_FIELDS = ["reason", "topic", "materiality", "direction", "forward_looking"]


def classify(row, item_id, model=MODEL):
    """Classify one change. Never raises for model or server problems: returns a record with
    status "ok" and the labels, or status "failed" and the error, so a long run keeps going."""

    record = {"item_id": item_id, "model": model, "prompt_version": PROMPT_VERSION,
              "status": "failed", "error": None, "raw": None,
              **{f: None for f in LABEL_FIELDS},
              "prompt_tokens": None, "output_tokens": None, "seconds": None}
    
    start = time.time()

    try:
        response = ollama.chat(
            model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_message(row)}],
                    format=ChangeLabel.model_json_schema(),
                    think=False,
                    options=OPTIONS)

        record["raw"] = response.message.content
        record["prompt_tokens"] = response.prompt_eval_count
        record["output_tokens"] = response.eval_count

        if response.done_reason == 'length':
            record["error"] = "hit num_predict"
        else:
            label = ChangeLabel.model_validate_json(response.message.content)
            record.update(label.model_dump())
            record["status"] = "ok"

    except ValidationError as e:
        record["error"] = f"schema: {e.errors()[0]['msg']}"
    except (ollama.ResponseError, ConnectionError) as e:
        record["error"] = f"ollama: {e}"
    record["seconds"] = round(time.time() - start, 2)

    return record


def results_path(model, items_name):
    """One file per model, prompt version and item set, so configurations never mix."""
    slug = model.replace(":", "-").replace("/", "-")
    return OUT_DIR / f"{slug}_{PROMPT_VERSION}_{items_name}.jsonl"


def read_results(path):
    """All records in a results file; the last record per item_id wins (a retry replaces a
    failure). A half-written last line from a crash is skipped."""
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[rec["item_id"]] = rec
    return records


def run(items, item_ids, path, model=MODEL):
    """Classify every item not yet done (status ok) in `path`, appending one line per item."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {i for i, r in read_results(path).items() if r["status"] == "ok"}
    todo = [(row, iid) for (_, row), iid in zip(items.iterrows(), item_ids) if iid not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {path.relative_to(ROOT)}")

    start = time.time()

    with path.open("a") as f:
        for k, (row, iid) in enumerate(todo, 1):
            record = classify(row, iid, model)
            f.write(json.dumps(record) + "\n")
            f.flush()

            if record['status'] != 'ok' or k % 25 == 0 or k == len(todo):
                time_left = (time.time() - start) / k * (len(todo) - k)
                print(f"  {k}/{len(todo)}  {record['status']:6s} {record['error'] or ''}"
                      f"  ~{time_left / 60:.0f} min left", flush=True)


def load_items(which, split=None, tickers=None):
    """(items DataFrame, item_ids, name for the results file)."""
    if which == "gold":
        items = pd.read_csv(GOLD_ITEMS_PATH)
        if split:
            items = items[items.split == split]
        ids = items.item_id
        name = f"gold_{split or 'all'}"
    else:
        items = pd.read_csv(CHANGES_PATH)
        items = items[items.change.isin(LABELLED_CHANGES)]
        if tickers:
            items = items[items.ticker.isin(tickers)]
        ids = [make_item_id(r) for r in items.itertuples()]   # same ids as the gold set uses
        name = "all" if not tickers else "all_" + "-".join(tickers)
    return items.reset_index(drop=True), list(ids), name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", choices=["gold", "all"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], help="gold only")
    parser.add_argument("--tickers", nargs="+", help="all only")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.items == "gold" and args.tickers or args.items == "all" and args.split:
        parser.error("--split goes with --items gold, --tickers with --items all")

    items, ids, name = load_items(args.items, args.split, args.tickers)
    path = results_path(args.model, name)
    run(items, ids, path, args.model)

    # Resuming makes partial runs normal, so report the state of the whole file, not this run
    records = [r for r in read_results(path).values() if r["item_id"] in set(ids)]
    failed = [r for r in records if r["status"] != "ok"]
    print(f"{len(records) - len(failed)} ok, {len(failed)} failed, "
          f"{len(ids) - len(records)} never attempted, of {len(ids)} items")
    for r in failed[:10]:
        print(f"  failed: {r['item_id']}  {r['error']}")
    if failed:
        print("Rerun the same command to retry the failed items.")


if __name__ == "__main__":
    main()