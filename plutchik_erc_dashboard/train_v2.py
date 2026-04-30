"""
Plutchik ERC v2.1 — Antigravity Training Harness
Supports FP16, Macro-F1 Checkpointing, and CSV-based ingestion.
"""

import torch
from torch.utils.data import DataLoader
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

# Add project to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from models.multitask_emotion_model import PluTchikMultiTaskModel, MultiTaskLoss
from utils.preprocessing import build_dataset_from_csv
from utils.trainer import PluTchikTrainer

# ============== CONFIGURATION ==============
CONFIG = {
    "csv_path": "data/processed/ERC/plutchik_v2_production.csv",
    "model_dir": "plutchik_erc_dashboard/my_plutchik_model",
    "batch_size": 8,
    "epochs": 5,          # Production run: 5 epochs (GRL needs room to activate)
    "lr": 2e-5,
    "max_len": 256,
    "iaa_weighting": True,
    "adv_weight": 0.3,    # Reduced from 1.0 to prevent "Total Erasure"
    "warmup_epochs": 1,   # 1 epoch warmup, then GRL activates for remaining 4
    "grl_lambda_max": 0.5 # Cap the lambda to prevent gradient explosion
}

# ============== PLUTCHIK CONSTANTS ==============
from utils.constants import PLUTCHIK, NUM_EMOTIONS

def run_antigravity_training():
    print("=" * 60)
    print("🚀 ANTIGRAVITY TRAINING HARNESS — PLUTCHIK ERC v2.1")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Target Device: {device}")
    
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
    
    # 2. Setup DataLoaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 3. Model Initialization
    print("\n🔧 Initializing RoBERTa-base Multi-Task Model...")
    model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)
    
    # 4. Loss Function (IAA Aware)
    loss_fn = MultiTaskLoss(
        emotion_weight=1.0,
        sarcasm_weight=0.7,
        intensity_weight=0.5,
        adv_weight=CONFIG["adv_weight"],
        iaa_weighting=CONFIG["iaa_weighting"]
    )
    
    # 5. Trainer (Hardware-Aware with FP16)
    trainer = PluTchikTrainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        save_dir=CONFIG["model_dir"]
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
    print(f"Best model saved to: {CONFIG['model_dir']}/best_model.pt")
    print("=" * 60)

if __name__ == "__main__":
    run_antigravity_training()
