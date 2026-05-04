#!/usr/bin/env python3
"""Count JSONL rows with human_verified=true (Stage 3 CDA gate)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="Path to contrastive_pairs*.jsonl")
    args = ap.parse_args()
    path: Path = args.jsonl
    if not path.is_file():
        print(f"0 verified / 0 total (missing file: {path})")
        return
    total = 0
    verified = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            if json.loads(line).get("human_verified"):
                verified += 1
        except json.JSONDecodeError:
            continue
    print(f"human_verified=True: {verified} / total lines: {total}")


if __name__ == "__main__":
    main()
