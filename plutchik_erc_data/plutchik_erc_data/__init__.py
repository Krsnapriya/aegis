"""Small data package: load versioned Plutchik ERC CSV from disk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

Split = Literal["train", "val", "test"]


def default_csv_path() -> Path:
    env = os.environ.get("PLUTCHIK_ERC_CSV", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    legacy = here.parents[2] / "data" / "processed" / "ERC" / "plutchik_v2_production.csv"
    return legacy


def load_dataset(split: Optional[Split] = None, csv_path: Optional[Path] = None) -> pd.DataFrame:
    path = csv_path or default_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"Plutchik ERC CSV not found: {path}")
    df = pd.read_csv(path)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    return df
