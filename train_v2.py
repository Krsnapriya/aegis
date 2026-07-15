"""
Plutchik ERC v2.1 — Antigravity Training Harness
Supports FP16, Macro-F1 Checkpointing, and CSV-based ingestion.
"""

import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import sys
from pathlib import Path
import json
import random
import numpy as np

# Bug 6 Fix: Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Project directory
project_dir = Path(__file__).parent

from models.multitask_emotion_model import PluTchikMultiTaskModel, MultiTaskLoss
from utils.preprocessing import build_dataset_from_csv, load_contrastive_pairs, PlutchikERCDataset
from utils.trainer import PluTchikTrainer

# ============== CONFIGURATION ==============
CONFIG = {
    "csv_path": "data/processed/ERC/plutchik_v2_production.csv",
    "model_dir": "my_plutchik_model",
    "batch_size": 4,      # Reduced to 4 to fix MPS OOM on M1
    "epochs": 1,          # Fast prototype generation
    "lr": 5e-5,
    "max_len": 128,       # Faster processing
    "iaa_weighting": True,
    "adv_weight": 0.3,    
    "warmup_epochs": 0,   
    "grl_lambda_max": 0.5 
}

# ============== PLUTCHIK CONSTANTS ==============
from utils.constants import PLUTCHIK, NUM_EMOTIONS, EMOTION_NAMES

def run_antigravity_training():
    print("=" * 60)
    print("🚀 ANTIGRAVITY TRAINING HARNESS — PLUTCHIK ERC v2.1")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n✓ Target Device: {device}")
    
    # Build idx_to_emotion mapping
    idx_to_emotion = {i: e for i, e in enumerate(EMOTION_NAMES)}
    
    # 1. Build Dataset from Production CSV
    print(f"\n📊 Ingesting production data from {CONFIG['csv_path']}...")
    csv_abs_path = Path(CONFIG["csv_path"]).resolve()
    if not csv_abs_path.exists():
        csv_abs_path = project_dir.parent / "data" / "processed" / "ERC" / "plutchik_v2_production.csv"
    
    # Bug 1 Fix: Load train and val separately based on the 'split' column
    train_ds = build_dataset_from_csv(
        str(csv_abs_path), 
        PLUTCHIK,
        tokenizer_name="roberta-base",
        split="train"
    )
    val_ds = build_dataset_from_csv(
        str(csv_abs_path), 
        PLUTCHIK,
        tokenizer_name="roberta-base",
        split="val"
    )
    
    print(f"✓ Train set: {len(train_ds)} samples")
    print(f"✓ Val set: {len(val_ds)} samples")

    cda_path = os.environ.get("PLUTCHIK_CDA_JSONL", "").strip()
    cda_min = int(os.environ.get("PLUTCHIK_CDA_MIN_PAIRS", "200"))
    if cda_path:
        candidates = [Path(cda_path), project_dir / cda_path, project_dir.parent / cda_path]
        cda_file = next((p for p in candidates if p.is_file()), None)
        if cda_file is not None:
            cda_samples = load_contrastive_pairs(str(cda_file), PLUTCHIK, only_verified=True)
            if len(cda_samples) >= cda_min:
                train_ds = ConcatDataset([train_ds, PlutchikERCDataset(cda_samples)])
                print(f"✓ Merged {len(cda_samples)} human-verified CDA rows (gate ≥{cda_min}).")
            else:
                print(
                    f"⚠ PLUTCHIK_CDA_JSONL set but only {len(cda_samples)} verified rows "
                    f"(need ≥{cda_min}). Skipping CDA merge — export more via pair_verifier."
                )
        else:
            print(f"⚠ CDA file not found (tried): {cda_path}")
    
    # 2. Setup DataLoaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 3. Model Initialization
    print("\n🔧 Initializing RoBERTa-base Multi-Task Model...")
    model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)
    
    # 4. Loss Function (IAA Aware + Wheel-Distance Weighted)
    loss_fn = MultiTaskLoss(
        emotion_weight=1.0,
        sarcasm_weight=0.7,
        intensity_weight=0.5,
        adv_weight=CONFIG["adv_weight"],
        iaa_weighting=CONFIG["iaa_weighting"],
        wheel_distance_weighting=True,
        idx_to_emotion=idx_to_emotion
    )
    
    # 5. Trainer (Hardware-Aware with FP16)
    _model_dir = Path(CONFIG["model_dir"])
    if not _model_dir.is_absolute():
        _model_dir = project_dir / _model_dir
    trainer = PluTchikTrainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        save_dir=str(_model_dir),
    )
    
    # 6. Execute Training
    print(f"\n🔥 Commencing training for {CONFIG['epochs']} epochs...")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        warmup_epochs=CONFIG["warmup_epochs"],
        grl_lambda_max=CONFIG["grl_lambda_max"]
    )
    
    # 7. Finalize
    print("\n" + "=" * 60)
    print("✨ TRAINING COMPLETE")
    print(f"Best model saved to: {_model_dir / 'best_model.pt'}")
    print("=" * 60)

if __name__ == "__main__":
    run_antigravity_training()
