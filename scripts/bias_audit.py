#!/usr/bin/env python3
"""
Per-scenario sarcasm F1 on the validation split (Stage 1 bias audit / post-DAT re-audit).
Writes routing_decision.json next to the checkpoint when --write-json is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "plutchik_erc_dashboard"
sys.path.insert(0, str(DASH))

from models.multitask_emotion_model import PluTchikMultiTaskModel  # noqa: E402
from utils.constants import NUM_EMOTIONS, PLUTCHIK  # noqa: E402
from utils.preprocessing import build_dataset_from_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "data/processed/ERC/plutchik_v2_production.csv")
    ap.add_argument("--checkpoint", type=Path, default=DASH / "my_plutchik_model/best_model.pt")
    ap.add_argument("--write-json", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = build_dataset_from_csv(str(args.csv), PLUTCHIK, split="val")
    model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS).to(device).eval()
    if args.checkpoint.is_file():
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
    else:
        print(f"⚠ No checkpoint at {args.checkpoint} — metrics will reflect random weights.")

    by_scenario: dict[str, dict[str, list]] = {}
    for raw in val_ds.samples:
        scen = raw.get("scenario", "unknown")
        by_scenario.setdefault(scen, {"y": [], "pred": []})
        ids = raw["input_ids"].unsqueeze(0).to(device)
        mask = raw["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(ids, mask, alpha=0.0)
        sp = torch.softmax(out["sarcasm_logits"], dim=-1)[0, 1].item()
        pred = int(sp >= 0.5)
        y = int(raw["sarcasm_label"].item()) if hasattr(raw["sarcasm_label"], "item") else int(raw["sarcasm_label"])
        by_scenario[scen]["y"].append(y)
        by_scenario[scen]["pred"].append(pred)

    rows = []
    for scen, d in sorted(by_scenario.items()):
        y, p = np.array(d["y"]), np.array(d["pred"])
        if y.sum() == 0 and (1 - y).sum() == 0:
            f1 = None
        elif y.sum() == 0 or (1 - y).sum() == 0:
            f1 = float((p == y).mean())
        else:
            f1 = float(f1_score(y, p, average="binary", zero_division=0))
        rows.append({"scenario": scen, "n": len(y), "sarcasm_positive_rate": float(y.mean()), "sarcasm_f1": f1})

    macro = np.nanmean([r["sarcasm_f1"] for r in rows if r["sarcasm_f1"] is not None])
    print(json.dumps({"per_scenario": rows, "macro_sarcasm_f1": float(macro)}, indent=2))

    gaps = []
    f1s = [r["sarcasm_f1"] for r in rows if r["sarcasm_f1"] is not None]
    if f1s:
        mx, mn = max(f1s), min(f1s)
        gaps.append({"sarcasm_f1_max_min_gap": float(mx - mn)})

    out_payload = {"per_scenario": rows, "macro_sarcasm_f1": float(macro), "gaps": gaps}
    if args.write_json:
        out_path = args.checkpoint.parent / "routing_decision.json"
        out_path.write_text(json.dumps(out_payload, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
