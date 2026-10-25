# FilingDelta plan (revised end of Day 1)

Focus: move fast through data plumbing; spend most of the time on the AI parts (embeddings, LLM classification, RAG, agent, evaluation).

## Day 1 (done)
- Repo, venv, config; `src/ingest.py` pulls 6 10-Qs + 2 10-Ks per ticker and saves MD&A.
- 11 tickers: JPM, MS, GOOGL, MSFT, NVDA, XOM, CVX, COP, WMT, PG, KO. BAC/WFC/C/GS/PNC/USB dropped (edgartools section parsing fails silently on their 10-Qs).
- Diff sanity check on MSFT with difflib: positional diff can't handle moved or slightly edited paragraphs, so order-independent matching is needed.

## Day 2: plumbing, fast (Claude writes, Prithvi reviews)
1. Manifest: save filing metadata (form, acceptance time, period, fiscal quarter, accession) with each file; decide pairing rule.
2. Validation: flag sections that are too short, too long, or don't start like MD&A. Eyeball every flagged file.
3. `src/clean.py`: paragraphs, drop page numbers and repeated headers, re-join page-split paragraphs, keep section headings as labels, tag tables.
4. `src/diff.py`: order-independent paragraph matching (mask numbers/dates, rapidfuzz similarity, one-to-one assignment), output a table of changes: section, old, new, type (unchanged / numbers-only / reworded / moved / added / removed), similarity. Hand-check ~20 rows; notes seed the Day 4 gold set.
5. Prices: daily prices (yfinance) for the 11 tickers, SPY and sector ETFs (XLF: JPM, MS; XLK: MSFT, NVDA; XLC: GOOGL; XLE: XOM, CVX, COP; XLP: WMT, PG, KO). Market model over days -250 to -30; CARs over [0,+1], [-1,+1], [0,+5]. Record earnings dates too: 10-Qs often come after the earnings release, so the price reaction may already have happened.

## Day 3: embeddings (first AI piece)
- Embed paragraphs; replace fuzzy matching with semantic matching; compare the two.
- Store embeddings (simple vector index) for later retrieval.

## Day 4: LLM change classification + evaluation
- For each changed paragraph pair: topic, materiality, direction (more cautious / more optimistic), with structured output.
- Hand-label ~50 pairs as a gold set; measure LLM accuracy; prompt iterations tracked.

## Day 5: RAG
- Question answering over filings with citations ("how has NVDA's discussion of supply constraints changed over 4 quarters?").
- Retrieval evaluation (did it fetch the right paragraphs?).

## Day 6: agent
- Given a ticker: fetch latest filing, diff, retrieve history, write a cited "what changed" report.

## Day 7: does it predict anything?
- Relate change scores to abnormal returns; out-of-sample; explain which change types matter.

## Day 8: responsible AI + write-up
- Explainability, failure cases, limitations; README, webpage, resume bullet.

## Backlog
- Heading-based MD&A fallback for big banks; possible edgartools issue/PR.
- Risk Factors year-over-year diff from 10-Ks.
- Scale to 40-60 companies and 5+ years for the signal test.
