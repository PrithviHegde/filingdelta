# Annotation guidelines: MD&A change classification

These guidelines define the gold labels for the LLM change classifier (`src/classify.py`).
They are the same rules the classifier's system prompt states (prompt v6), so that the model
and the annotators are judged against one definition. If a rule here changes, the prompt
changes with it, and its version number goes up.

## What is being labelled

One row of `data/processed/changes_embedding.csv` whose `change` is `reworded`, `added` or
`removed`: a paragraph of a company's MD&A compared with its counterpart in an earlier filing
(same quarter last year, previous quarter, or previous 10-K). Number-only updates
(`numbers_only`) and identical paragraphs (`unchanged`) are never labelled.

The question for every field: **from the point of view of an investor, what did the wording
change say?** Numbers are context for how large a change is, never a reason on their own.

## Procedure

1. Label **blind**: read OLD and NEW and decide before looking at any model output, score or
   other annotator's label. (A label made after seeing the tool's answer inherits its blind
   spots: the first diff hand check marked three paragraphs "added" or "removed" because the
   tool showed no partner, and they had one.)
2. Use only the paragraph pair, ticker, periods and section heading, as the model does. Don't
   look up the filing for extra context.
3. Fill in `reason` first, then the four labels.
4. Two flags instead of a label when needed:
   - `bad_match`: OLD and NEW are not versions of the same paragraph (a matching error from
     `src/diff.py`, e.g. a JPM lead-in sentence "...consisted of:" paired with a summary
     sentence). Leave the labels empty; these rows are counted, not scored.
   - `ambiguous`: you can defend two labels for a field. Pick the better one, flag it, and say
     why in `note`. Ambiguous rows show how hard the task is; they are not errors.

## Fields

### reason
One sentence, at most 25 words, quoting the changed words. For materiality 0, say why
nothing changed ("cross-reference added", "six-month comparison added; same explanation").

### materiality

Start from 0. Move up only if you can name what an investor learns from NEW that OLD did not
say.

| Value | Meaning | Examples |
|---|---|---|
| 0 | Nothing new | see the list below |
| 1 | New detail that doesn't change the picture | an extra driver named; a clarification; a small restatement |
| 2 | Changes the picture | a new or removed risk; changed guidance; a reversed trend ("decreased" -> "increased"); a new business driver; a new legal or regulatory matter; changed hedging about the future ("expect" -> "may") |

**Always 0:**
- the explanation is unchanged and only numbers, dates or periods changed, including an added
  year-to-date (six- or nine-month) comparison next to the quarter (every Q2 and Q3 10-Q does this)
- cross-references ("refer to Note X", "see pages Y"), whatever they point to
- a dropped footnote or caveat that no longer applies (e.g. a footnote explaining a rating
  outlook that has since been resolved)
- grammar, punctuation, word order and formatting; synonyms that don't change the meaning
  ("further" -> "substantial advancements" in otherwise identical text)

A reversal in the direction of a named revenue or cost line counts as a reversed trend (2),
regardless of the line's size.

**2 is strict.** Only a new or removed risk, changed guidance or outlook, a reversed trend,
a new legal or regulatory matter, or a major event (such as a geopolitical disruption to
operations). New or changed drivers, extra detail and acquisitions the company calls not
material are 1.

**Superseded paragraphs are 0.** When a period-specific paragraph is removed or added only
because the new filing carries its own version of it (last year's "net revenues decreased 5%"
sentence, a prior-year dividend declaration, a year-to-date paragraph a Q2 filing adds), label
materiality 0 and flag `ambiguous` with the note "superseded; likely matching miss". The
change itself tells an investor nothing.

### direction

Is the company's situation or outlook described as better or worse than before?

| Value | Use when |
|---|---|
| `more_positive` | the change reads better for the company (growth replaces decline, a risk is gone, stronger language) |
| `more_negative` | the change reads worse (decline replaces growth, a new risk, weaker or more hedged language) |
| `no_clear_direction` | the change matters but has no clear sign: a new factual detail, a changed driver that is neither better nor worse, or positive and negative elements together |
| `not_applicable` | materiality 0 only |

- Materiality 0 -> always `not_applicable`; materiality 1 or 2 -> never `not_applicable`. A change
  that matters but reads neither better nor worse is `no_clear_direction`.
- This value was first called `neutral_or_mixed` (renamed with prompt v6, same meaning): the old
  name suggested only the "mixed" case, and models gave a sign to most unsigned details.
- Added paragraphs: judge the new information. Removed paragraphs: judge the information dropped.
  A removed risk or problem is usually `more_positive`; an added one `more_negative`.

### topic

The main subject of the changed content. If several fit, choose the one most important to an
investor.

| Value | Covers |
|---|---|
| `revenue_demand` | sales, volumes, pricing, customer demand, segment or product revenue |
| `costs_margins` | operating expenses, cost of goods sold, gross margins, SG&A, interest expense, cost savings |
| `outlook_guidance` | **only** explicit company guidance or targets; otherwise use the subject the statement is about |
| `capital_liquidity` | debt, cash, buybacks, dividends, capital ratios, **credit ratings of the company**, funding |
| `credit_risk` | loan losses, allowances, charge-offs, credit quality **of loans** (mainly the banks) |
| `legal_regulatory` | litigation, new compliance rules, government investigations, regulatory matters |
| `macro` | inflation, interest rates, foreign exchange, supply chains, broader economic or industry trends (incl. the energy transition) |
| `strategy_mna` | mergers, acquisitions, divestitures, strategic shifts, joint ventures, restructuring, major capital projects and cost-reduction programmes (e.g. Willow, LNG projects, synergy targets) |
| `accounting` | accounting policies, restatements, critical accounting estimates, revenue recognition |
| `other` | anything else, incl. business descriptions, **cross-references** and boilerplate |

Every row gets a topic, including materiality-0 rows: the subject of the paragraph, even when
the change is trivial (an added "and" in a revenue bullet is `revenue_demand`). Use `other`
only when no category fits. Cross-references are always `other`.

### forward_looking

`true` if the paragraph is about the future (expectations, guidance, plans, risks that may
happen); `false` if it only describes the past or present. Judge the whole paragraph (NEW, or
OLD if removed), not only the words that changed, and set it for materiality-0 rows too.

Why the paragraph and not the changed words: it is easier to label consistently (a comma edit
in an outlook paragraph has no tense of its own), and for the material changes that matter in
the analysis the two readings almost always agree.

## Worked examples

From the smoke-test items, which are not in the gold set.

| Item | Change | Label | Why |
|---|---|---|---|
| MS qoq, Fixed Income underwriting | "decreased primarily reflecting lower bond and investment-grade loan issuances" -> "increased primarily reflecting higher ... issuances, which benefitted from higher event-related activity" | revenue_demand · 2 · more_positive · false | reversed trend in a named revenue line, plus a new driver |
| NVDA yoy, added | "For descriptions of our facility lease guarantees, long-term debt ... refer to Note 8, Note 9 ..." | other · 0 · not_applicable · false | cross-reference |
| KO qoq, interest expense | Q2 filing adds the six-month comparison; drivers ("higher debt balances, partially offset by lower rates on derivative instruments") unchanged | costs_margins · 0 · not_applicable · false | numbers and period structure only |
| JPM yoy, removed | footnote explaining Moody's "developing" outlook dropped after the outlook moved to stable | capital_liquidity · 0 · not_applicable · false | dropped caveat that no longer applies; ratings are capital_liquidity (ambiguous: removed_information with a positive direction is defensible) |
| CVX yoy, climate outlook | "transition to a lower carbon future will progress" -> "a lower carbon future progresses", "further" -> "substantial advancements", two commas; rest identical | macro · 0 · not_applicable · true | wording polish, same meaning; the paragraph is about the future |
| JPM qoq, credit | "The net addition to the allowance for credit losses consisted of:" vs "The provision for credit losses was $9.6 billion. Net charge-offs ..." | `bad_match` | a lead-in sentence matched to a different paragraph |

## Gold set

- ~50 rows sampled with a fixed seed, stratified by change type, pair type and ticker;
  smoke-test and `labels/diff_handcheck.csv` rows excluded.
- Split before labelling: ~35 **dev** rows (used to compare models and prompts) and ~15
  **test** rows (scored once, at the end).
- Two annotators label independently (blind to each other); agreement is reported per field
  (Cohen's kappa); disagreements are adjudicated with a one-line reason; the adjudicated label
  is the gold label.
