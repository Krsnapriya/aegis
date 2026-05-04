#!/usr/bin/env python3
"""
Export PluTchikMultiTaskModel to ONNX (opset 14) for Stage 5 compression path.
Requires: pip install onnx onnxscript
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "plutchik_erc_dashboard"
sys.path.insert(0, str(DASH))

from models.multitask_emotion_model import PluTchikMultiTaskModel  # noqa: E402
from utils.constants import NUM_EMOTIONS  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Single forward returning tensors ONNX can register as outputs."""

    def __init__(self, inner: PluTchikMultiTaskModel):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        out = self.inner(input_ids, attention_mask, alpha=0.0)
        return (
            out["emotion_logits"],
            out["sarcasm_logits"],
            out["intensity"].squeeze(-1),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DASH / "my_plutchik_model/best_model.pt")
    ap.add_argument("--out", type=Path, default=DASH / "my_plutchik_model/plutchik_multitask.onnx")
    args = ap.parse_args()

    device = "cpu"
    model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS).to(device).eval()
    if args.checkpoint.is_file():
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
    wrapped = ExportWrapper(model).eval()

    dummy_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_mask = torch.ones(1, 128, dtype=torch.long)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask),
        str(args.out),
        input_names=["input_ids", "attention_mask"],
        output_names=["emotion_logits", "sarcasm_logits", "intensity"],
        opset_version=14,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "emotion_logits": {0: "batch"},
            "sarcasm_logits": {0: "batch"},
            "intensity": {0: "batch"},
        },
    )
    print(f"Exported ONNX to {args.out}")


if __name__ == "__main__":
    main()
