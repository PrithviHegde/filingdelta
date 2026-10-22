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
notebooks/01_explore_edgar.ipynb            exploration of EDGAR filings and section extraction, with findings
src/config.py                               tickers, forms, SEC identity
src/ingest.py                               ingestion pipeline: fetch filings, extract MD&A, save
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
python -m src.ingest
```

This downloads the last 6 10-Qs and 2 10-Ks for each ticker, extracts MD&A, and writes one text file per filing period to `data/processed/`. Filings that fail are logged and skipped, and a saved / skipped summary is printed per ticker.

## Known issues and limitations

- **Silent parser failures on large banks.** For BAC, WFC, C, GS, PNC and USB, `edgartools` section extraction either can't find 10-Q MD&A or returns the wrong section (e.g. BAC's share repurchase table in place of MD&A; WFC's financial statement notes). WFC's 10-K MD&A is in Exhibit 13. These banks are excluded for now, so the bank sector is thin. Details in `notebooks/01_explore_edgar.ipynb`.
- **"Saved" does not mean "correct."** A validation step (minimum length, MD&A heading check) is planned.
- **Page numbers and headers** remain in the extracted text and need cleaning before diffing.
- **Fiscal calendars differ** (NVDA, WMT, KO, PG), so filings are aligned by filing date / fiscal quarter rather than calendar quarter.
- **MD&A length grows through the fiscal year** (Q1 shortest), need to check if this is actually a real trend, or just an anomaly. If it is, it has to be normalized.

## Roadmap

- [x] Ingest MD&A for 11 companies × 8 quarters
- [ ] Validation step for extracted sections
- [ ] Text cleaning (page numbers, headers)
- [ ] Paragraph-level diffing between quarters
- [ ] Separate meaningful language changes from number updates
- [ ] Stock price data and event-study setup
- [ ] RAG / LLM layer to explain what changed
- [ ] Evaluation and explainability
