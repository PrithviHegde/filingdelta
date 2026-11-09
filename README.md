# FilingDelta

**Research question:** Do changes in the language of SEC filings (10-Qs and 10-Ks) show up in stock prices, and with what lag?

FilingDelta pulls company filings from SEC EDGAR, extracts the sections where management's language changes from quarter to quarter, measures what changed, and tests whether those changes predict stock moves.

> Status: work in progress.

## Approach

- **MD&A, quarter over quarter.** Management's Discussion & Analysis is extracted from every 10-Q and 10-K (Item 2 in a 10-Q, Item 7 in a 10-K). This is the main, frequent signal.
- **Risk Factors, year over year.** 10-Q risk factor sections are usually just a pointer to the annual report, so risk factor changes are tracked from 10-Ks, annually.
- There is no Q4 10-Q: the 10-K covers Q4, so 8 quarters = 6 10-Qs + 2 10-Ks.

Why section extraction matters: a single JPMorgan 10-Q is ~1.2M characters (~300k tokens), mostly tables and boilerplate. MD&A is ~22% of it.

## Companies

11 large caps across four sectors:

| Sector | Tickers |
|---|---|
| Banks | JPM, MS |
| Tech / AI | GOOGL, MSFT, NVDA |
| Energy | XOM, CVX, COP |
| Consumer | WMT, PG, KO |

The ticker list is in `src/config.py`.

## Project structure

```
data/processed/{ticker}/{period}_mdna.txt   extracted MD&A text (not committed; regenerate with the pipeline)
data/processed/{ticker}/{period}_mdna_clean.txt   cleaned MD&A, one block per paragraph, for reading (raw file left untouched)
data/processed/paragraphs.csv               one row per paragraph, all filings: kind (text / heading / table), section, text
data/processed/manifest.csv                 one row per filing: form, accession, acceptance time, fiscal year / quarter, file path, validation flags
data/processed/pairs.csv                    which filing is compared with which (yoy, qoq, annual)
data/processed/changes.csv                  one row per matched / added / removed paragraph per pair, with change label
labels/diff_handcheck.csv                   24 hand-annotated changes.csv rows (label correct? + note), seed for the gold set
notebooks/01_explore_edgar.ipynb            exploration of EDGAR filings and section extraction, with findings
notebooks/02_diff_sanity.ipynb              difflib baseline on MSFT (why paragraph matching must ignore order)
notebooks/03_explore_diff.ipynb             building blocks of diff.py on one MSFT pair
src/config.py                               tickers, forms, SEC identity
src/ingest.py                               ingestion pipeline: fetch filings, extract MD&A, save, write manifest
src/validate.py                             flag MD&A that is too short, too long or doesn't start like MD&A
src/clean.py                                clean MD&A and split it into labelled paragraphs
src/pairs.py                                build pairs.csv from the manifest
src/diff.py                                 match paragraphs between paired filings and label the changes
```

## Setup

Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

SEC EDGAR requires a name and email with every request. Set `SEC_IDENTITY` in `src/config.py` to your own.

## Running

From the project root:

```bash
python -m src.ingest                  # download + extract MD&A for all tickers, write manifest
python -m src.ingest --manifest-only  # MD&A already on disk: fetch metadata only, write manifest
python -m src.ingest --tickers MSFT   # one ticker
python -m src.validate                # add validation flags to the manifest (rerun after every ingest)
python -m src.clean                   # clean MD&A, write paragraphs.csv
python -m src.pairs                   # build pairs.csv from the manifest
python -m src.diff                    # match paragraphs per pair, write changes.csv (~1 min)
python -m src.diff --tickers MSFT     # some tickers: print a report, don't write
```

`src.ingest` downloads the last 6 10-Qs and 2 10-Ks for each ticker, extracts MD&A, and writes one text file per filing period to `data/processed/`, plus `manifest.csv` with each filing's metadata. Filings that fail are logged and skipped, and a saved / skipped summary is printed per ticker.

`src.pairs` pairs filings on fiscal year / quarter: **yoy** (10-Q vs the same fiscal quarter last year, the primary comparison since it removes seasonal wording), **qoq** (10-Q vs the previous 10-Q in the same fiscal year), and **annual** (10-K vs prior 10-K). Amendments are excluded. The current window gives 78 pairs: 33 yoy, 34 qoq, 11 annual.

`src.diff` compares the text paragraphs of each pair (tables left out: their changes are number updates). Numbers, percentages, $ amounts, dates and period words ("three months", "second quarter") are masked first, so "revenue grew 17%" → "20%" is a number update, not a language change. Every old paragraph is scored against every new one (RapidFuzz `token_sort_ratio`, case-insensitive), then paired one-to-one with optimal assignment (Hungarian algorithm), so a paragraph that moved still finds its partner. Labels: **unchanged** (same text), **numbers_only** (same after masking), **reworded**, **added**, **removed**. Result: 18,636 rows over 78 pairs; a median 95% of old paragraphs find a partner in qoq pairs, 82% in yoy, 84% in annual.

## Known issues and limitations

- **Silent parser failures on large banks.** For BAC, WFC, C, GS, PNC and USB, `edgartools` section extraction either can't find 10-Q MD&A or returns the wrong section (e.g. BAC's share repurchase table in place of MD&A; WFC's financial statement notes). WFC's 10-K MD&A is in Exhibit 13. These banks are excluded for now, so the bank sector is thin. Details in `notebooks/01_explore_edgar.ipynb`.
- **"Saved" does not mean "correct."** `src.validate` flags MD&A under 3,000 words, over 2× the ticker's 10-Q median (5× for 10-Ks, which run 0.9-3.9× it), or without an MD&A heading in its first 500 characters. Current run: 2 of 88 flagged, both JPM 10-Ks, real MD&A that opens with ~800 words of financial highlights; reviewed and kept as valid, with the flag on record.
- **MD&A extraction runs into Item 8 for CVX, JPM and XOM 10-Ks.** edgartools' Item 7 continues into management's report on internal control and the auditor's report (4-22% of the text). None of the validation checks catches it. `src.clean` cuts at the first standalone Item 8 heading; without that, changing audit-report wording would show up as language changes in annual pairs.
- **Cleaning is rule-based, so it has known misses.** `src.clean` drops page numbers, footers and running headers (short lines that repeat *next to a page break*; a first version that dropped every repeated short line also removed real segment headings and number-only sentences), re-joins paragraphs cut by a page break (only when the cut is clearly mid-sentence: the second half starts lowercase, or the first ends in a lowercase word or comma), and tags tables. XOM's 2026 10-Qs come out of edgartools hard-wrapped (a blank line after every printed line); the join rule puts them back together, and their paragraph lengths now match XOM's other 10-Qs. Known misses: a few real headings that usually sit at a page top are dropped (JPM "Quarterly results", XOM "Volume / Mix", KO/PG "Nine months ended ... versus ..."); some tables made mostly of column headers stay tagged as text; a split whose second half starts with "(" or whose first half ends in a capitalised word stays as two paragraphs. Result: 26,091 paragraphs (19,305 text, 4,661 headings, 2,125 tables).
- **Fiscal calendars differ** (MSFT and PG end in June, NVDA and WMT in January, KO and NVDA use 52/53-week years), so filings are aligned by fiscal quarter, derived from SEC's fiscal year end, never by calendar date.
- **Filing date is not the event date.** SEC gives filings accepted after 5:30pm ET the next business day's filing date (all 8 GOOGL filings, one PG). Event timing uses the acceptance time from the filing header instead, with a 4pm cutoff for the next trading day.
- **Earnings confound.** Some companies (e.g. MSFT, GOOGL, NVDA) file the 10-Q on the evening of their earnings release, so the filing's event day is also the earnings reaction day. Others file weeks after earnings (JPM, MS, WMT file 2-4 weeks after their quarter's results). Earnings dates need to be recorded per filing and controlled for.
- **Small sample.** The 8-filing window gives only 3 yoy pairs per ticker (33 in total). Raising the 10-Q window is under consideration.
- **Fuzzy matching has a precision / recall trade-off with no clean cut-off.** Best-match scores don't split into two clear groups: 60-80 mixes real rewrites with look-alike template sentences (MSFT's "X increased $N or N% driven by Y" under every segment, JPM's "Equity Markets revenue" vs "Fixed Income Markets revenue"). Thresholds were set by reading samples per score band: a match needs 65 within the same section, 80 across sections, and same-section partners get a small tie-break bonus. Some wrong partners remain in the 65-80 band (roughly 1 in 4 there), which inflates "reworded" and hides some added / removed paragraphs. This is the baseline that embeddings are compared against.
- **Hand check of 24 change rows** (stratified by label, 9 tickers; 10 reworded, half of them from the 65-80 score band): 23 labels correct. The miss is a wrong partner at score 66 (JPM: a CECL footnote matched to an unrelated capital-ratio footnote). "Reworded" means the wording changed, not that the meaning did: some are grammar knock-ons of period updates (WMT "for both the three and six months" → "the three and nine months"). Judging materiality is left to the change classifier.
- **No "moved" label.** A flag for "matched paragraph now under a different heading" was tried and dropped: section labels from clean.py are too noisy (renamed headings, a "Table of Contents" section, headings attached to the wrong paragraphs), and all 3 flags in the hand check were wrong. JPM alone had 391, because its 10-Qs repeat sentences under both "Quarterly results" and "Year-to-date results".
- **MD&A length grows through the fiscal year** (Q1 shortest), need to check if this is actually a real trend, or just an anomaly. If it is, it has to be normalized.

## Roadmap

- [x] Ingest MD&A for 11 companies × 8 quarters
- [x] Filing manifest (acceptance time, fiscal quarter) and pairing rule: 78 pairs
- [x] Validation step for extracted sections
- [x] Text cleaning: Item 8 trim, page junk, page-split paragraphs, section labels, table tags
- [x] Paragraph-level diffing between quarters (order-independent matching)
- [x] Separate meaningful language changes from number updates
- [ ] Stock price data and event-study setup
- [ ] RAG / LLM layer to explain what changed
- [ ] Evaluation and explainability
