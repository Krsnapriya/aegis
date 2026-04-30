"""
Plutchik ERC — Flywheel Training Automator
Closes the loop: DB Corrections -> Training Data -> Fine-tuned Model.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models.db_models import DB_Correction
from ..models.multitask_emotion_model import PluTchikMultiTaskModel, MultiTaskLoss
from .trainer import PluTchikTrainer
from .preprocessing import build_dataset_from_csv
from .constants import PLUTCHIK, NUM_EMOTIONS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flywheel-trainer")

def run_flywheel_update(model_dir="plutchik_erc_dashboard/my_plutchik_model", min_samples=5):
    """
    1. Extract reviewed corrections from DB.
    2. Convert to training format.
    3. Run a focused fine-tuning pass.
    """
    db = SessionLocal()
    try:
        # Fetch reviewed corrections
        reviewed = db.query(DB_Correction).filter(DB_Correction.status == "reviewed").all()
        
        if len(reviewed) < min_samples:
            logger.info(f"Insufficient new samples ({len(reviewed)}). Need {min_samples} to trigger flywheel.")
            return False

        logger.info(f"🔥 Triggering Flywheel Update with {len(reviewed)} new samples...")

        # 1. Create temporary training data
        data = []
        for c in reviewed:
            data.append({
                "text": c.text,
                "emotion": c.corrected_emotion,
                "sarcasm": 0, # Default for corrections
                "intensity": 0.5, # Default for corrections
                "scenario": "hitl_correction",
                "split": "train", # Force into training set
                "iaa_weight": 2.0 # Give HITL corrections high weight
            })
        
        df_new = pd.DataFrame(data)
        temp_csv = "data/processed/ERC/flywheel_temp.csv"
        os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
        df_new.to_csv(temp_csv, index=False)

        # 2. Initialize Model & Weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)
        
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded base weights from {checkpoint_path}")

        # 3. Prepare DataLoader
        train_ds = build_dataset_from_csv(temp_csv, PLUTCHIK, split="train")
        train_loader = DataLoader(train_ds, batch_size=min(len(train_ds), 8), shuffle=True)

        # 4. Setup Loss & Trainer
        loss_fn = MultiTaskLoss(iaa_weighting=True)
        trainer = PluTchikTrainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
            save_dir=model_dir
        )

        # 5. Fine-tune (High speed, low epochs for flywheel)
        logger.info("Starting fine-tuning pass...")
        trainer.fit(
            train_dataloader=train_loader,
            val_dataloader=train_loader, # Validate on same small set for flywheel
            epochs=3,
            learning_rate=1e-5,
            warmup_epochs=0
        )
        
        # 6. Mark as ingested
        for c in reviewed:
            c.status = "ingested"
        db.commit()
        
        logger.info("✓ Flywheel update complete. Model weights updated.")
        
        # Cleanup
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
        return True

    except Exception as e:
        logger.error(f"Flywheel Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    run_flywheel_update()
