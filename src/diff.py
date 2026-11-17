"""Compare the MD&A paragraphs of each filing pair and label what changed.

For every pair in data/processed/pairs.csv:
  1. take the old and new filing's text paragraphs from data/processed/paragraphs.csv
  2. mask()               hide numbers, percentages, $ amounts and dates
  3. score every old paragraph against every new one, with one of two methods:
       fuzzy      similarity_matrix(): RapidFuzz token_sort_ratio on masked text (0-100)
       embedding  embedding_scores(): cosine of bge-small embeddings of the masked text,
                  directions masked too (0-1; vectors from src/embed.py)
  4. match()              pair old and new paragraphs one-to-one, in any order
  5. label()              unchanged / numbers_only / reworded for each match;
                          old paragraphs left unmatched are "removed", new ones "added"

Masking is what separates number updates ("revenue grew 17%" -> "revenue grew 20%") from
real language changes. Order-independent matching is what difflib couldn't do (notebook 02):
a paragraph that moved or was lightly edited still finds its partner.

The two methods differ only in step 3 and its thresholds; masking for labels, matching
and labelling are shared, so their outputs can be compared row for row (notebook 04).

Writes data/processed/changes.csv (fuzzy) or changes_embedding.csv: one row per matched
pair, added or removed paragraph.

Run from the project root:
    python -m src.diff                              # fuzzy, all pairs, write changes.csv
    python -m src.diff --method embedding           # write changes_embedding.csv
    python -m src.diff --tickers MSFT               # some tickers: print only, no write
"""

import argparse
import re
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz, utils
from scipy.optimize import linear_sum_assignment

from src.config import ROOT, DATA_DIR

PAIRS_PATH = DATA_DIR / "pairs.csv"
PARAGRAPHS_PATH = DATA_DIR / "paragraphs.csv"
CHANGES_PATH = {"fuzzy": DATA_DIR / "changes.csv",
                "embedding": DATA_DIR / "changes_embedding.csv"}
# Written by src/embed.py: row k of the vectors is row k of the index
EMBEDDINGS_INDEX_PATH = DATA_DIR / "embeddings_index.csv"
EMBEDDINGS_MASKED_PATH = DATA_DIR / "embeddings_masked.npy"

# Which paragraph kinds are compared. Tables are left out: their changes are number updates.
COMPARE_KINDS = ["text"]

LABELS = ["unchanged", "numbers_only", "reworded", "added", "removed"]


# ---------------------------------------------------------------------------
# Comparison: mask, score, match, label
# ---------------------------------------------------------------------------

def mask(text):
    """Return text with numbers, percentages, $ amounts, dates and period words replaced
    by placeholders, so that number-only updates compare as equal.

    E.g. "Revenue increased $5.1 billion or 17% in the three months ended March 31, 2026"
      -> "Revenue increased <XX> or <XX> in the <PERIOD> months ended <XX>"
    """

    # 1. Dates: Matches MM/DD/YYYY, YYYY-MM-DD, written months (Oct 25, 2026), etc.
    date_pattern = (
        r"\b\d{1,4}[./-]\d{1,4}[./-]\d{2,4}\b|"
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{2,4}\b"
    )

    # 2. Currency: Matches $100, £50.25, $1,000.00, etc.
    currency_pattern = r"[$£¥€]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion)\b)?"

    # 3. Percentages: Matches 50%, 12.5%, etc.
    percent_pattern = r"\b\d+(?:\.\d+)?%"

    # 4. General Numbers: Standalone digits, decimals, numbers with commas
    number_pattern = r"\b\d+(?:,\d{3})*(?:\.\d+)?\b"

    # 5. Period words: "three months ended" -> "six months ended", "first quarter" -> "second
    #    quarter". These change mechanically every quarter (qoq pairs especially), like dates,
    #    so they shouldn't count as rewording. Masked as <PERIOD>, not <XX>, so "the <XX>
    #    months" (a number) and "the <PERIOD> months" stay distinguishable if ever needed.
    period_pattern = (
        r"\b(?:three|six|nine|twelve)(?=[\s-]+months?\b)|"
        r"\b(?:first|second|third|fourth)(?=[\s-]+(?:fiscal\s+)?quarters?\b)"
    )
    text = re.sub(period_pattern, "<PERIOD>", text, flags=re.IGNORECASE)

    # Combine all patterns using the OR (|) operator
    combined_pattern = (
        f"{date_pattern}|{currency_pattern}|{percent_pattern}|{number_pattern}"
    )

    return re.sub(combined_pattern, "<XX>", text, flags=re.IGNORECASE)

# Direction words. A first list from reading MSFT / JPM / XOM text, not complete
# ("improved", "deteriorated", "up", "down" are not in it).
DIRECTION = r"\b(?:increase[sd]?|decrease[sd]?|grew|grow|declined?|higher|lower|rose|fell|gains?|loss(?:es)?|(?:un)?favorabl[ey]|positively|negatively)\b"


def mask_direction(text):
    """Return text with direction words ("increased", "lower", ...) replaced by <DIR>.

    Used for matching only, on top of mask(); never for labelling. Matching asks "is this
    the same paragraph?", and a flipped direction doesn't change that. Without it, the
    embedding matched "Xbox ... revenue increased 8%" to "Azure ... revenue increased 40%"
    (0.872) over "Xbox ... revenue decreased 5%" (0.805): in a short sentence the verb
    outweighed the subject. Labelling must still see the flip, so label() compares mask()
    text, and "increased" -> "decreased" stays "reworded".

    Side effect: very short template sentences become identical across segments ("Revenue
    <DIR> <XX> or <XX>%."), so the same-section tie-break in match() decides between them.
    """
    return re.sub(DIRECTION, "<DIR>", text, flags=re.IGNORECASE)


def similarity_matrix(old_masked, new_masked):
    """Return a 2-D array of scores, shape (len(old_masked), len(new_masked)), 0-100.
    scores[i, j] = how similar old paragraph i is to new paragraph j (after masking).
    old_masked and new_masked are paragraph texts, which have all numbers mask()ed
    """
    # default_process lowercases and strips punctuation first: without it a recapitalised
    # phrase ("condensed consolidated" -> "Condensed Consolidated", XOM) scored only 66.
    return process.cdist(old_masked, new_masked, scorer=fuzz.token_sort_ratio,
                         processor=utils.default_process)


class EmbeddingStore:
    """The masked paragraph vectors from src/embed.py, loaded once and looked up by
    (accession_no, para_no). Vectors are normalized, so a dot product is the cosine."""

    def __init__(self):
        index = pd.read_csv(EMBEDDINGS_INDEX_PATH, dtype={"accession_no": str})
        self.vectors = np.load(EMBEDDINGS_MASKED_PATH)
        self.row_of = {(a, p): k for k, (a, p) in enumerate(zip(index["accession_no"], index["para_no"]))}

    def vectors_for(self, paras, accession_no):
        # A KeyError means a paragraph diff.py compares was never embedded: rerun src.embed
        return self.vectors[[self.row_of[(accession_no, p["para_no"])] for p in paras]]

    def scores(self, old, new, old_accession_no, new_accession_no):
        """Cosine matrix, shape (len(old), len(new)), 0-1."""
        return self.vectors_for(old, old_accession_no) @ self.vectors_for(new, new_accession_no).T


# Minimum score for a match. Chosen by reading best-match pairs by score band (MSFT, JPM,
# XOM, KO) and using "both paragraphs sit under the same heading" as a rough precision check:
# the same-section rate is 33% in the 50s, 59% in the 60s, 74% in the 70s, 84-92% above 80
# (it never reaches 100%: headings get renamed). The 50s are almost all wrong partners.
# 60-80 holds real rewrites but also template look-alikes (MSFT's "X increased $N or N%
# driven by Y" bullets). Within a section, reading 10 samples per band over 6 tickers:
# 60-65 about half wrong (JPM "Equity Markets revenue" <-> "Fixed Income Markets revenue"),
# 65-70 ~7/10 right, 70-75 ~9/10 right. Across sections (read on MSFT) nearly all 70-80
# matches were wrong, e.g. one segment's "Cost of revenue increased..." matched to
# another segment's. So: 65 within a section, 80 across.
# Remaining wrong partners are the known weakness of fuzzy matching, the baseline that
# embedding-based matching is compared against.
#
# Tie-break toward partners under the same heading, used only inside the assignment (the
# reported score stays raw). Short template sentences ("Revenue increased $<XX> or <XX>%.")
# are identical after masking under every MSFT segment, so without it the solver splits
# ties across segments arbitrarily (one segment's sentence matched to another's). Small
# enough that a paragraph that really moved (its true partner scores ~100 under the new
# heading) still wins.
FUZZY = {"min_same_section": 65, "min_other_section": 80, "section_bonus": 5}

# Embedding thresholds, set the same way: one-to-one matches on MSFT, JPM, XOM, KO, the
# same-section rate per 0.02 cosine band, then ~8 samples read per band (none from the
# hand check in labels/, which is kept for evaluation).
# - Same-section rate: ~20% below 0.76, 43-51% at 0.76-0.82, then a step to 71% at 0.84
#   and 83-95% above 0.92. Fuzzy scores had no such step.
# - Same section: 0.76-0.84 about half right (JPM lead-in sentences matched to their own
#   bullets, debt vs equity underwriting); 0.84 5/8; 0.86 7/8 -> 0.86.
# - Other section: 0.84-0.88 3/8, 0.88-0.92 2/8, 0.92-0.96 ~4/7 right, and no better
#   higher up -> 0.92. Above that the score stops separating right from wrong: the errors
#   are segment templates (XOM "Margin -" in Chemical vs Energy, MSFT opex in two
#   segments), the correct ones mostly renamed or noisy headings. So cross-section
#   precision tops out around 60%, a limit of the section labels, not of the score.
# - Bonus 0.01: the fuzzy +5 rescaled to the compressed cosine range (most scores 0.8-1.0).
# Small samples (8 per band): judgement calls, tested on the hand check, not tuned on it.
EMBEDDING = {"min_same_section": 0.86, "min_other_section": 0.92, "section_bonus": 0.01}


def match(scores, same_section, min_same_section, min_other_section, section_bonus):
    """Return a list of (i, j) pairs: old paragraph i matched with new paragraph j.

    scores:        score matrix, shape (n_old, n_new): fuzzy (0-100) or cosine (0-1)
    same_section:  bool array, same shape: old i and new j sit under the same heading
    thresholds and bonus are on the scale of `scores` (FUZZY or EMBEDDING above)

    Each i and each j appears at most once (one-to-one). A paragraph with no good partner
    is left out, and becomes "removed" (old) or "added" (new).

    Optimal assignment (Hungarian algorithm) rather than greedy: greedy takes the highest
    score first, which can give a boilerplate paragraph the partner another paragraph
    needed more. Optimal maximises the total score over all pairs. On clear matches
    (score ~100, most of them) the two agree; they differ only on ambiguous ones.

    Pairs below the threshold are zeroed *before* the assignment, not just filtered after:
    otherwise the solver could accept a worse eligible pair to make room for an
    ineligible one that we then throw away.
    """
    eligible = np.where(same_section, scores >= min_same_section, scores >= min_other_section)
    allowed = np.where(eligible, scores + section_bonus * same_section, 0.0)
    rows, cols = linear_sum_assignment(allowed, maximize=True)
    # The solver always returns min(n_old, n_new) pairs; drop the ones it was forced into
    return [(int(i), int(j)) for i, j in zip(rows, cols) if eligible[i, j]]


def normalize(text):
    """Collapse whitespace, so a reflowed line break doesn't count as a change."""
    return " ".join(text.split())


def label(old, new):
    """Return "unchanged", "numbers_only" or "reworded" for a matched pair.

    old and new are dicts with text (as filed) and masked (mask(text)).

    Decided on exact text, not on the score: token_sort_ratio ignores word order, so a
    score of 100 doesn't prove the text is identical, and any threshold below 100 would
    hide one-word edits ("expect" -> "may"), which are exactly what we're after.

    There is no "moved" label: section labels from clean.py are too noisy for it (a
    24-row hand check found all 3 sampled "moved" flags were renamed or mislabelled
    headings). Sections are still used in match(), where noise only weakens a tie-break.
    """
    if normalize(old["text"]) == normalize(new["text"]):
        return "unchanged"
    if normalize(old["masked"]) == normalize(new["masked"]):
        return "numbers_only"
    return "reworded"


def same_heading(a, b):
    """Headings equal after masking: "Fiscal Year 2025 Compared with Fiscal Year 2024" and
    "Fiscal Year 2026 Compared with ... 2025" are the same section in consecutive filings."""
    return a is not None and b is not None and normalize(mask(a)).lower() == normalize(mask(b)).lower()


# ---------------------------------------------------------------------------
# Plumbing: loading, looping over pairs, writing changes.csv
# ---------------------------------------------------------------------------

def filing_paragraphs(paragraphs, accession_no):
    """The comparable paragraphs of one filing, in filing order, as a list of dicts."""
    rows = paragraphs[(paragraphs["accession_no"] == accession_no)
                      & paragraphs["kind"].isin(COMPARE_KINDS)]
    rows = rows.sort_values("para_no")
    return [
        {
            "para_no": r.para_no,
            "text": r.text,
            "section": None if pd.isna(r.section) else r.section,
        }
        for r in rows.itertuples()
    ]


def load_pair(pair, paragraphs):
    """(old paragraphs, new paragraphs) for one row of pairs.csv. Also used in notebook 03.

    Each paragraph is a dict with para_no, text, section (not yet masked).
    """
    return (filing_paragraphs(paragraphs, pair["old_accession_no"]),
            filing_paragraphs(paragraphs, pair["new_accession_no"]))


def diff_pair(pair, paragraphs, method="fuzzy", store=None):
    """All change rows for one pair: matched paragraphs, then removed, then added.

    method "fuzzy" or "embedding"; store is an EmbeddingStore (needed for "embedding").
    """
    old, new = load_pair(pair, paragraphs)
    for p in old + new:
        p["masked"] = mask(p["text"])   # for label(): numbers masked, directions kept
    if method == "fuzzy":
        scores = similarity_matrix([p["masked"] for p in old], [p["masked"] for p in new])
        thresholds, decimals = FUZZY, 1
    else:
        scores = store.scores(old, new, pair["old_accession_no"], pair["new_accession_no"])
        thresholds, decimals = EMBEDDING, 3
    same_section = np.array([[same_heading(o["section"], n["section"]) for n in new] for o in old],
                            dtype=bool).reshape(len(old), len(new))
    matches = match(scores, same_section, **thresholds)

    base = {"ticker": pair["ticker"], "pair_type": pair["pair_type"],
            "old_period": pair["old_period_of_report"], "new_period": pair["new_period_of_report"],
            "old_accession_no": pair["old_accession_no"], "new_accession_no": pair["new_accession_no"]}

    def row(change, o, n, score):
        return {**base, "change": change, "score": score,
                "old_para_no": o["para_no"] if o else None, "new_para_no": n["para_no"] if n else None,
                "old_section": o["section"] if o else None, "new_section": n["section"] if n else None,
                "old_text": o["text"] if o else None, "new_text": n["text"] if n else None}

    rows = [row(label(old[i], new[j]), old[i], new[j], round(float(scores[i, j]), decimals))
            for i, j in matches]
    matched_old = {i for i, _ in matches}
    matched_new = {j for _, j in matches}
    rows += [row("removed", o, None, None) for i, o in enumerate(old) if i not in matched_old]
    rows += [row("added", None, n, None) for j, n in enumerate(new) if j not in matched_new]
    return rows


def print_report(changes, n_examples=2, n_chars=160):
    """Label counts per pair type, then a few examples of each label to read by eye."""
    counts = changes.groupby(["pair_type", "change"]).size().unstack(fill_value=0)
    print(counts.reindex(columns=[c for c in LABELS if c in counts.columns]).to_string())

    for change in LABELS:
        examples = changes[changes["change"] == change].head(n_examples)
        for _, r in examples.iterrows():
            print(f"\n--- {change} ({r['ticker']} {r['pair_type']} {r['old_period']} -> {r['new_period']}, score {r['score']})")
            if pd.notna(r["old_text"]):
                print(f"  OLD: {r['old_text'][:n_chars]}")
            if pd.notna(r["new_text"]):
                print(f"  NEW: {r['new_text'][:n_chars]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="only these tickers: print, don't write")
    parser.add_argument("--method", choices=["fuzzy", "embedding"], default="fuzzy",
                        help="paragraph scoring (default fuzzy, the baseline)")
    args = parser.parse_args()

    pairs = pd.read_csv(PAIRS_PATH)
    paragraphs = pd.read_csv(PARAGRAPHS_PATH)
    if args.tickers:
        pairs = pairs[pairs["ticker"].isin(args.tickers)]
    store = EmbeddingStore() if args.method == "embedding" else None

    rows = []
    for _, pair in pairs.iterrows():
        rows.extend(diff_pair(pair, paragraphs, args.method, store))
    changes = pd.DataFrame(rows)
    print_report(changes)

    out = CHANGES_PATH[args.method]
    if args.tickers:
        print(f"\n--tickers given: {out.name} not written")
    else:
        changes.to_csv(out, index=False)
        print(f"\nWrote {len(changes)} rows to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
