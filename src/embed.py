"""Embed every text paragraph with BAAI/bge-small-en-v1.5, in two versions:

  embeddings_masked.npy  mask_direction(mask(text)): numbers, dates and directions hidden,
                         for paragraph matching in src/diff.py
  embeddings_raw.npy     text as filed, for retrieval (numbers matter in answers)

embeddings_index.csv gives the paragraph (accession_no, para_no) for each row of both.

Run from the project root:
    python -m src.embed                  # all tickers, write the three files
    python -m src.embed --tickers MSFT   # some tickers: check only, no write
"""

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR
from src.diff import mask, COMPARE_KINDS, mask_direction

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PARAGRAPHS_PATH = DATA_DIR / "paragraphs.csv"
INDEX_PATH = DATA_DIR / "embeddings_index.csv"
MASKED_PATH = DATA_DIR / "embeddings_masked.npy"
RAW_PATH = DATA_DIR / "embeddings_raw.npy"


def load_text_paragraph(tickers=None) -> pd.DataFrame:
    """loads the paragraphs.csv file, filters to text paragraphs, and optionally filters to given tickers.
    This is the input for the embedding model"""

    df = pd.read_csv(PARAGRAPHS_PATH, dtype={"accession_no": str, "para_no": int, "kind": str, "text": str, "ticker": str})
    paragraphs = df[df['kind'].isin(COMPARE_KINDS)].sort_values(["accession_no", "para_no"]).reset_index(drop=True)
    if tickers is not None:
        paragraphs = paragraphs[paragraphs['ticker'].isin(tickers)].sort_values(["accession_no", "para_no"]).reset_index(drop=True)

    return paragraphs


def embed(texts, model: SentenceTransformer) -> np.ndarray:
    """Return embeddings for a list of texts."""
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+")
    args = parser.parse_args()

    paragraphs = load_text_paragraph(args.tickers)
    model = SentenceTransformer(MODEL_NAME)

    masked = embed(paragraphs['text'].apply(lambda t: mask_direction(mask(t))).tolist(), model)
    raw = embed(paragraphs['text'].tolist(), model)

    # Checks: one vector per paragraph, all normalized, no NaNs
    for name, arr in [("masked", masked), ("raw", raw)]:
        assert arr.shape == (len(paragraphs), 384), f"{name}: shape {arr.shape}"
        assert np.allclose(np.linalg.norm(arr, axis=1), 1, atol=1e-4), f"{name}: not normalized"
        assert not np.isnan(arr).any(), f"{name}: NaNs"

    print(f"{len(paragraphs)} paragraphs, arrays {masked.shape}")

    if args.tickers:
        print("--tickers given: nothing written")
    else:
        np.save(MASKED_PATH, masked.astype(np.float32))
        np.save(RAW_PATH, raw.astype(np.float32))
        # Row k of the index = row k of both arrays
        paragraphs[["accession_no", "para_no", "ticker"]].to_csv(INDEX_PATH, index=False)
        print(f"Wrote {MASKED_PATH.name}, {RAW_PATH.name}, {INDEX_PATH.name}")


if __name__ == "__main__":
    main()