"""Does the classified MD&A language line up with stock returns? Three pre-specified tests.

The sample is 56 filings (11 tickers, 2 years; the first filing of each ticker has nothing to
compare with), so nothing is fitted: the scores and tests below were fixed before any return
was looked at. With nothing fitted there is nothing to overfit, which is why there is no
train/test split. The price of that: only fairly large correlations can be detected (|rho| of
about 0.26 on 56 filings, 0.33 on the 36 not confounded by an earnings release).

Each filing is scored on one comparison: a 10-Q against the same quarter last year (removes
seasonal wording), a 10-K against the previous 10-K, and a 10-Q with no year-ago filing in the
window against the previous quarter. Only changes the classifier rated materiality 2 are used,
split in two by a rule-based detector:
- trend: the lead sentence's direction word flipped ("decreased" -> "increased"), or its
  growth word changed size ("low single digits" -> "mid-single digits", which is how PG reports
  segment sales and which mask() cannot hide). These restate the reported numbers.
- other: the rest rated 2, minus those about revenue or costs (topic revenue_demand or
  costs_margins). Reading samples showed the revenue/cost ones are mostly the reported numbers
  again in another form ("net earnings increased 6%" -> "increased 2%", table rows), so what is
  left is language about risk, outlook, legal matters, strategy, capital, credit and the economy.
  Decided after reading the changes, before looking at any return.
Tone of a group = (more_positive - more_negative) / (more_positive + more_negative), from the
classifier's direction label; 0 when the group is empty.

Hypotheses (primary sample and window in brackets):
- H1, sanity check [all filings; earnings-release SCAR, 0 to +1]: trend tone is positively
  related to the earnings-release return. The trend changes restate the results the market
  already reacted to at the release, so this tests whether the labels measure something real.
- H2, the research question [filings not confounded by earnings; filing SCAR, 0 to +1]: other
  tone is positively related to the return on the filing itself.
- H3, size not sign [same]: the share of paragraphs with a material change is positively
  related to the size of the filing-day move, |SCAR|. Needs no direction label, the
  classifier's weakest field.

Returns: SCAR from src/event_study.py (market model vs SPY), CAR divided by its standard error,
so tickers with different volatility are comparable. "Not confounded" = no earnings release in
the window (earnings_in_window False): MSFT, GOOGL, NVDA and COP 10-Qs file on release day.

Statistic: Spearman correlation after demeaning both variables within ticker (PG has far more
material changes than JPM, which says nothing about returns), with a p-value from a permutation
test that shuffles scores within ticker. Eleven tickers are too few clusters for clustered
standard errors; the permutation test needs no such assumption.

Run from the project root:
    python -m src.price_test              # features, tests, figures
"""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.classify import load_items, read_results, results_path, MODEL
from src.config import DATA_DIR, ROOT

PAIRS_PATH = DATA_DIR / "pairs.csv"
PARAGRAPHS_PATH = DATA_DIR / "paragraphs.csv"
EVENTS_PATH = DATA_DIR / "events.csv"
FEATURES_PATH = DATA_DIR / "filing_features.csv"
FIG_DIR = ROOT / "docs" / "figures"

NUMBER_TOPICS = ["revenue_demand", "costs_margins"]
N_PERMUTATIONS = 10_000
SEED = 7

# Direction words, matched on the lead of the paragraph (its first direction word): MD&A
# sentences open with the line item's move ("Operating income decreased $316 million ...") and
# list drivers after it ("driven by higher pricing"), so the first match is the headline move.
UP = r"increas\w*|higher|grew|rose|improv\w*"
DOWN = r"decreas\w*|lower|declin\w*|fell"
FLAT = r"unchanged|flat"
DIRECTION = re.compile(rf"\b(?:(?P<up>{UP})|(?P<down>{DOWN})|(?P<flat>{FLAT}))\b", re.I)
# PG / KO style growth words, ordered by size
SIZE = re.compile(r"\b(low|mid|high)[- ]single[- ]digits?|\b(double)[- ]digits?", re.I)
SIZE_RANK = {"low": 1, "mid": 2, "high": 3, "double": 4}


def lead_move(text):
    """Signed size of the paragraph's headline move: +1/-1/0 for up/down/flat, times the size
    of a 'low/mid/high single digits' phrase right after it (1 if there is none). None if the
    paragraph has no direction word."""
    if not isinstance(text, str):
        return None
    m = DIRECTION.search(text)
    if not m:
        return None
    sign = 1 if m["up"] else -1 if m["down"] else 0
    size = SIZE.match(text[m.end():].lstrip())
    rank = SIZE_RANK[(size[1] or size[2]).lower()] if size else 1
    return sign * rank


def trend_change(old_text, new_text):
    """+1 / -1 if the headline move turned up / down between OLD and NEW (a flipped direction
    or a changed growth size), 0 if it did not or a side has no direction word."""
    old, new = lead_move(old_text), lead_move(new_text)
    if old is None or new is None or old == new:
        return 0
    return int(np.sign(new - old))


def load_changes():
    """Classified changes with their text, pair and labels (status ok only)."""
    items, ids, name = load_items("all")
    recs = read_results(results_path(MODEL, name))
    labels = pd.DataFrame([recs.get(i, {"status": "missing"}) for i in ids])
    df = pd.concat([items, labels[["status", "topic", "materiality", "direction"]]], axis=1)
    df = df[df.status == "ok"].copy()
    df["materiality"] = df.materiality.astype(int)
    df["trend"] = [trend_change(o, n) if c == "reworded" else 0
                   for o, n, c in zip(df.old_text, df.new_text, df.change)]
    return df


def tone(directions):
    pos, neg = (directions == "more_positive").sum(), (directions == "more_negative").sum()
    return (pos - neg) / (pos + neg) if pos + neg else 0.0


def filing_features(changes):
    """One row per filing that has a comparison: the chosen pair and the scores."""
    pairs = pd.read_csv(PAIRS_PATH)
    # One comparison per filing: yoy / annual where there is one, else qoq
    pairs["rank"] = pairs.pair_type.map({"yoy": 0, "annual": 0, "qoq": 1})
    chosen = pairs.sort_values("rank").drop_duplicates("new_accession_no")

    paras = pd.read_csv(PARAGRAPHS_PATH, usecols=["accession_no", "kind"])
    n_text = paras[paras.kind == "text"].groupby("accession_no").size()

    rows = []
    for p in chosen.itertuples():
        c = changes[(changes.new_accession_no == p.new_accession_no)
                    & (changes.old_accession_no == p.old_accession_no)]
        m2 = c[c.materiality == 2]
        is_trend = m2.trend != 0
        is_other = ~is_trend & ~m2.topic.isin(NUMBER_TOPICS)
        rows.append({
            "ticker": p.ticker, "accession_no": p.new_accession_no, "form": p.new_form,
            "period": p.new_period_of_report, "pair_type": p.pair_type,
            "n_text_paragraphs": n_text.get(p.new_accession_no, np.nan),
            "n_changes": len(c), "n_material": len(m2),
            "n_trend": int(is_trend.sum()), "n_other": int(is_other.sum()),
            "material_share": len(m2) / n_text.get(p.new_accession_no, np.nan),
            "trend_tone": tone(m2[is_trend].direction),
            "trend_tone_rule": float(np.sign(m2[is_trend].trend.sum())),   # detector's own sign
            "other_tone": tone(m2[is_other].direction),
        })
    return pd.DataFrame(rows)


def add_returns(features):
    """SCARs (market model) for the filing and its earnings release, [0,+1] and [0,+5]."""
    ev = pd.read_csv(EVENTS_PATH)
    ev = ev[ev.benchmark == "market"]
    out = features.copy()
    for event, window, col in [("filing", "[+0,+1]", "filing_scar"),
                               ("filing", "[+0,+5]", "filing_scar_5"),
                               ("earnings", "[+0,+1]", "earnings_scar")]:
        e = ev[(ev.event == event) & (ev.window == window)].set_index("accession_no")
        out[col] = out.accession_no.map(e.scar)
        if event == "filing":
            out[col.replace("scar", "confounded")] = out.accession_no.map(e.earnings_in_window)
    return out


def demean(s, groups):
    return s - s.groupby(groups).transform("mean")


def within_ticker_test(df, x, y, rng):
    """(n, raw Spearman, within-ticker Spearman, two-sided permutation p)."""
    d = df[[x, y, "ticker"]].dropna()
    xd, yd = demean(d[x], d.ticker), demean(d[y], d.ticker)
    rho = spearmanr(xd, yd).statistic
    groups = [np.flatnonzero(d.ticker.values == t) for t in d.ticker.unique()]
    xv, null = xd.values, np.empty(N_PERMUTATIONS)
    for k in range(N_PERMUTATIONS):
        perm = xv.copy()
        for g in groups:
            perm[g] = xv[rng.permutation(g)]
        null[k] = spearmanr(perm, yd).statistic
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1)
    return len(d), spearmanr(d[x], d[y]).statistic, rho, p


TESTS = [
    # (label, x, y, sample) -- the first row of each hypothesis is its pre-specified primary test
    ("H1 trend tone vs earnings SCAR", "trend_tone", "earnings_scar", "all"),
    ("H1 (detector's sign)", "trend_tone_rule", "earnings_scar", "all"),
    ("H2 other tone vs filing SCAR", "other_tone", "filing_scar", "clean"),
    ("H2 (all filings)", "other_tone", "filing_scar", "all"),
    ("H2 ([0,+5], clean)", "other_tone", "filing_scar_5", "clean5"),
    ("H3 material share vs |filing SCAR|", "material_share", "abs_filing_scar", "clean"),
    ("H3 (all filings)", "material_share", "abs_filing_scar", "all"),
    ("H3 ([0,+5], clean)", "material_share", "abs_filing_scar_5", "clean5"),
]


def run_tests(df):
    rng = np.random.default_rng(SEED)
    df = df.assign(abs_filing_scar=df.filing_scar.abs(), abs_filing_scar_5=df.filing_scar_5.abs())
    samples = {"all": df, "clean": df[~df.filing_confounded.astype(bool)],
               "clean5": df[~df.filing_confounded_5.astype(bool)]}
    rows = []
    for label, x, y, s in TESTS:
        n, raw, rho, p = within_ticker_test(samples[s], x, y, rng)
        rows.append({"test": label, "sample": s, "n": n, "spearman": raw,
                     "within_ticker_rho": rho, "p_perm": p})
    return pd.DataFrame(rows)


def plot(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = df.assign(abs_filing_scar=df.filing_scar.abs())
    clean = df[~df.filing_confounded.astype(bool)]
    panels = [(df, "trend_tone", "earnings_scar", "H1: trend tone vs earnings-release SCAR [0,+1]"),
              (clean, "other_tone", "filing_scar", "H2: other tone vs filing SCAR [0,+1], clean"),
              (clean, "material_share", "abs_filing_scar", "H3: material share vs |filing SCAR|, clean")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (d, x, y, title) in zip(axes, panels):
        xd, yd = demean(d[x], d.ticker), demean(d[y], d.ticker)
        ax.scatter(xd, yd, s=22, alpha=0.7)
        ax.axhline(0, lw=0.5, c="grey"); ax.axvline(0, lw=0.5, c="grey")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{x} (demeaned within ticker)"); ax.set_ylabel(f"{y} (demeaned within ticker)")
    fig.tight_layout()
    path = FIG_DIR / "price_tests.png"
    fig.savefig(path, dpi=130)
    return path


NULL_MEAN_ABS = np.sqrt(2 / np.pi)   # E|Z| for Z ~ N(0,1): mean |SCAR| on a day with no news


def news_on_filing_days(features):
    """Is there any price news on filing days? Under no news a SCAR is ~N(0,1), so mean |SCAR|
    is 0.80 and 5% exceed 1.96. Compared for earnings releases, filings in the same window as a
    release, and filings on their own. Added after H2 and H3 came back null (so exploratory),
    but it needs no labels and explains them: if filing days carry no news, nothing in the
    text can line up with that day's return."""
    groups = {"earnings release": features.earnings_scar,
              "filing, earnings in window": features.filing_scar[features.filing_confounded.astype(bool)],
              "filing on its own": features.filing_scar[~features.filing_confounded.astype(bool)]}
    rng = np.random.default_rng(SEED)
    rows = []
    for name, s in groups.items():
        a = s.dropna().abs().values
        boot = [rng.choice(a, len(a)).mean() for _ in range(N_PERMUTATIONS)]
        rows.append({"days": name, "n": len(a), "mean_abs_scar": a.mean(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "share_above_1.96": np.mean(a > 1.96)})
    return pd.DataFrame(rows), groups


def plot_news(groups):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    rng = np.random.default_rng(SEED)
    for y, (name, s) in enumerate(groups.items()):
        a = s.dropna().abs().values
        ax.scatter(a, y + rng.uniform(-0.15, 0.15, len(a)), s=18, alpha=0.6)
        ax.plot([a.mean()] * 2, [y - 0.3, y + 0.3], c="black", lw=2)
        ax.text(max(a.max(), 3) + 0.2, y, f"mean {a.mean():.2f}, {np.mean(a > 1.96):.0%} > 1.96",
                va="center", fontsize=8)
    ax.axvline(NULL_MEAN_ABS, ls="--", c="grey", lw=1)
    ax.axvline(1.96, ls=":", c="grey", lw=1)
    ax.set_yticks(range(len(groups)), [f"{k} (n={v.notna().sum()})" for k, v in groups.items()])
    ax.set_xlim(0, 13)
    ax.set_xlabel("|SCAR| [0,+1] vs SPY  (dashed: no-news mean 0.80; dotted: 1.96)")
    ax.set_title("Size of the price reaction: earnings releases vs 10-Q/10-K filings", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "filing_day_news.png"
    fig.savefig(path, dpi=130)
    return path


def main():
    changes = load_changes()
    feats = add_returns(filing_features(changes))
    feats.to_csv(FEATURES_PATH, index=False)
    print(f"{len(feats)} filings -> {FEATURES_PATH.relative_to(ROOT)}; "
          f"{(~feats.filing_confounded.astype(bool)).sum()} not confounded by earnings [0,+1]")
    m2 = changes[changes.materiality == 2]
    other = (m2.trend == 0) & ~m2.topic.isin(NUMBER_TOPICS)
    print(f"material changes: {len(m2)}, of which trend {(m2.trend != 0).sum()}, "
          f"other {other.sum()}, neither {len(m2) - (m2.trend != 0).sum() - other.sum()}")
    print(f"filings with no 'other' change: {(feats.n_other == 0).sum()}; "
          f"median per filing: trend {feats.n_trend.median():.0f}, other {feats.n_other.median():.0f}\n")
    res = run_tests(feats)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"({len(TESTS)} tests: Bonferroni threshold {0.05 / len(TESTS):.4f})\n")

    news, groups = news_on_filing_days(feats)
    print(f"Price news on filing days (no news: mean |SCAR| {NULL_MEAN_ABS:.2f}, 5% above 1.96)")
    print(news.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nfigures -> {plot(feats).relative_to(ROOT)}, {plot_news(groups).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
