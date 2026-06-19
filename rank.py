#!/usr/bin/env python3
"""
Redrob Hackathon — Ranking CLI.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces a top-100 CSV matching submission_spec.docx sections 2-3.
Runs fully locally: no network calls, no GPU. TF-IDF + rule-based scoring
only, so it comfortably fits the 5-minute / 16GB / CPU-only budget even
over the full 100K-candidate pool.
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.scoring import score_candidates, build_top100_rows  # noqa: E402

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.1f}s")

    t1 = time.time()
    results = score_candidates(candidates)
    print(f"Scored {len(results)} candidates in {time.time() - t1:.1f}s")

    rows = build_top100_rows(results)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total = time.time() - t0
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Total runtime: {total:.1f}s")


if __name__ == "__main__":
    main()
