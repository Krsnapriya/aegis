#!/usr/bin/env python3
"""
CSV hygiene: clamp emotion_cause to max words; optional semantic near-duplicate drop.
Semantic mode requires: pip install sentence-transformers scikit-learn
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plutchik_erc_dashboard"))
from utils.preprocessing import standardize_emotion_cause  # noqa: E402


def _semantic_dedupe(df: pd.DataFrame, text_col: str, thresh: float) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = df[text_col].astype(str).tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=True)
    keep = []
    banned = set()
    for i in range(len(df)):
        if i in banned:
            continue
        keep.append(i)
        sims = cosine_similarity(emb[i : i + 1], emb[i + 1 :])[0]
        for j, s in enumerate(sims, start=i + 1):
            if s >= thresh:
                banned.add(j)
    return df.iloc[sorted(keep)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--max-words", type=int, default=7)
    ap.add_argument("--semantic", action="store_true", help="Drop near-duplicates on text column")
    ap.add_argument("--semantic-thresh", type=float, default=0.95)
    ap.add_argument("--text-col", default="text")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "emotion_cause" in df.columns:
        df["emotion_cause"] = df["emotion_cause"].map(lambda x: standardize_emotion_cause(x, max_words=args.max_words))
    if args.semantic and args.text_col in df.columns:
        before = len(df)
        df = _semantic_dedupe(df, args.text_col, args.semantic_thresh)
        print(f"Semantic dedupe: {before} -> {len(df)} rows (thresh={args.semantic_thresh})")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
