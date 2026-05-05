"""
Training script for Plutchik Multi-Task Emotion Recognition Model.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import pickle
from sklearn.metrics import f1_score
from torch.amp import autocast
from torch.amp import GradScaler


class PluTchikTrainer:
    """
    Trainer for multi-task emotion recognition model.
    """
    
    def __init__(self, model, loss_fn, device: str = "cpu", 
                 save_dir: str = "./my_plutchik_model"):
        """
        Initialize trainer.
        
        Args:
            model: PluTchikMultiTaskModel instance
            loss_fn: MultiTaskLoss instance
            device: 'cpu' or 'cuda'
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.training_history = {
            "loss": [], "emotion_acc": [], "sarcasm_acc": [], "intensity_mse": [], "emotion_f1": [],
            "adv_loss": [], "disc_acc": [], "grad_norm": []
        }
        self.scaler = GradScaler("cuda", enabled=(device == "cuda"))
        self.global_step = 0
        self.total_steps = 0
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                   epoch: int, warmup_steps: int = 0, grl_lambda_max: float = 1.0) -> Dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            warmup_steps: Steps before GRL activates
            grl_lambda_max: Max value for GRL lambda (lambda_max)
        
        Returns:
            Epoch metrics dict
        """
        self.model.train()
        total_loss = 0.0
        emotion_correct = 0
        sarcasm_correct = 0
        intensity_mse = 0.0
        disc_correct = 0
        total_samples = 0
        total_grad_norm = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            emotion_labels = batch["emotion_label"].to(self.device)
            sarcasm_labels = batch["sarcasm_label"].to(self.device)
            intensity_labels = batch["intensity_label"].to(self.device)
            iaa_weights = batch["iaa_weight"].to(self.device)
            scenario_labels = batch["scenario_label"].to(self.device)
            
            # GRL Lambda Warmup Logic
            # p is progress from 0 to 1
            # alpha follows: 2 / (1 + exp(-10 * p)) - 1
            # But we only start increasing alpha AFTER the warmup_steps gate
            if self.global_step < warmup_steps:
                alpha = 0.0
            else:
                p = float(self.global_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                # Lambda follows: 2 / (1 + exp(-10 * p)) - 1, capped at grl_lambda_max
                alpha = (2. / (1. + np.exp(-10 * p)) - 1) * grl_lambda_max
            
            # Forward with autocast
            optimizer.zero_grad()
            with autocast("cuda", enabled=(self.device == "cuda")):
                # Handle optional Dissonance Head fields
                context_ids = batch.get("context_input_ids")
                if context_ids is not None:
                    context_ids = context_ids.to(self.device)
                    context_mask = batch["context_attention_mask"].to(self.device)
                    dissonance_labels = batch["dissonance_label"].to(self.device)
                else:
                    context_mask = None
                    dissonance_labels = None

                outputs = self.model(
                    input_ids, attention_mask, alpha=alpha,
                    context_input_ids=context_ids,
                    context_attention_mask=context_mask
                )
                
                # Loss
                targets = {
                    "emotion": emotion_labels,
                    "sarcasm": sarcasm_labels,
                    "intensity": intensity_labels,
                    "scenario": scenario_labels,
                    "dissonance": dissonance_labels
                }
                loss, loss_breakdown = self.loss_fn(outputs, targets, iaa_weights)
            
            # Milestone Checkpoint: Save exactly at the warmup gate transition
            if self.global_step == warmup_steps and warmup_steps > 0:
                milestone_path = os.path.join(self.save_dir, "checkpoint_pre_adversarial.pt")
                torch.save({
                    "step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "emotion_acc_at_gate": emotion_correct / max(1, total_samples)
                }, milestone_path)
                print(f"\n🛡️  Transition checkpoint saved to {milestone_path}")
                print(f"✓ Emotion Acc at gate: {emotion_correct / max(1, total_samples):.4f}")

            # Backward with scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            
            # Gradient Norm Logging (shared_dense layer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            
            self.scaler.step(optimizer)
            self.scaler.update()
            self.global_step += 1
            
            # Metrics
            batch_size = input_ids.shape[0]
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            
            # Emotion accuracy
            emotion_preds = outputs["emotion_logits"].argmax(dim=1)
            emotion_correct += (emotion_preds == emotion_labels).sum().item()
            
            # Sarcasm accuracy
            sarcasm_preds = outputs["sarcasm_logits"].argmax(dim=1)
            sarcasm_correct += (sarcasm_preds == sarcasm_labels).sum().item()
            
            # Intensity MSE
            intensity_mse += loss_breakdown["intensity_loss"] * batch_size
            
            # Discriminator Accuracy
            disc_preds = outputs["scenario_logits"].argmax(dim=1)
            disc_correct += (disc_preds == scenario_labels).sum().item()
            
            pbar.set_postfix({
                "loss": total_loss / total_samples,
                "em_acc": emotion_correct / total_samples,
                "alpha": alpha
            })
        
        metrics = {
            "loss": total_loss / total_samples,
            "emotion_accuracy": emotion_correct / total_samples,
            "sarcasm_accuracy": sarcasm_correct / total_samples,
            "intensity_mse": intensity_mse / total_samples,
            "disc_accuracy": disc_correct / total_samples,
            "grad_norm": total_grad_norm / len(dataloader),
            "last_lambda": alpha if 'alpha' in locals() else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict:
        """
        Validate on a dataset.
        
        Args:
            dataloader: Validation DataLoader
            epoch: Current epoch number
        
        Returns:
            Validation metrics dict
        """
        self.model.eval()
        total_loss = 0.0
        emotion_correct = 0
        sarcasm_correct = 0
        intensity_mse = 0.0
        total_samples = 0
        
        all_emotion_preds = []
        all_emotion_targets = []
        all_embeddings_by_emotion = {}  # For centroid calculation
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [VALID]", leave=False)
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            emotion_labels = batch["emotion_label"].to(self.device)
            sarcasm_labels = batch["sarcasm_label"].to(self.device)
            intensity_labels = batch["intensity_label"].to(self.device)
            iaa_weights = batch["iaa_weight"].to(self.device)
            scenario_labels = batch["scenario_label"].to(self.device)
            
            # Handle optional Dissonance Head fields
            context_ids = batch.get("context_input_ids")
            if context_ids is not None:
                context_ids = context_ids.to(self.device)
                context_mask = batch["context_attention_mask"].to(self.device)
                dissonance_labels = batch["dissonance_label"].to(self.device)
            else:
                context_mask = None
                dissonance_labels = None

            outputs = self.model(
                input_ids, attention_mask, alpha=0.0,
                context_input_ids=context_ids,
                context_attention_mask=context_mask
            )
            
            targets = {
                "emotion": emotion_labels,
                "sarcasm": sarcasm_labels,
                "intensity": intensity_labels,
                "scenario": scenario_labels,
                "dissonance": dissonance_labels
            }
            loss, _ = self.loss_fn(outputs, targets, iaa_weights)
            
            batch_size = input_ids.shape[0]
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            
            # Intensity MSE
            intensity_pred = outputs["intensity"].squeeze()
            intensity_mse += torch.nn.functional.mse_loss(intensity_pred, intensity_labels).item() * batch_size
            
            # Emotion predictions
            emotion_preds = outputs["emotion_logits"].argmax(dim=1)
            emotion_correct += (emotion_preds == emotion_labels).sum().item()
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_targets.extend(emotion_labels.cpu().numpy())
            
            # Sarcasm
            sarcasm_preds = outputs["sarcasm_logits"].argmax(dim=1)
            sarcasm_correct += (sarcasm_preds == sarcasm_labels).sum().item()
            
            # Store embeddings by emotion for centroid calculation
            cls_embeddings = outputs["cls_embedding"].cpu().numpy()
            for i, emotion_idx in enumerate(emotion_labels.cpu().numpy()):
                if emotion_idx not in all_embeddings_by_emotion:
                    all_embeddings_by_emotion[emotion_idx] = []
                all_embeddings_by_emotion[emotion_idx].append(cls_embeddings[i])
            
            pbar.set_postfix({
                "loss": total_loss / total_samples,
                "em_acc": emotion_correct / total_samples
            })
        
        # Compute F1 Score
        f1_macro = f1_score(all_emotion_targets, all_emotion_preds, average='macro')
        
        metrics = {
            "loss": total_loss / total_samples,
            "emotion_accuracy": emotion_correct / total_samples,
            "emotion_f1_macro": f1_macro,
            "sarcasm_accuracy": sarcasm_correct / total_samples,
            "intensity_mse": intensity_mse / total_samples,
            "embeddings_by_emotion": all_embeddings_by_emotion
        }
        
        return metrics
    
    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
           epochs: int = 5, learning_rate: float = 2e-5, warmup_epochs: int = 2,
           grl_lambda_max: float = 1.0) -> Dict:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            epochs: Number of epochs
            learning_rate: Learning rate for optimizer
            warmup_epochs: Number of epochs before GRL activations
            grl_lambda_max: Max value for GRL lambda
        
        Returns:
            Training history dict
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        self.total_steps = epochs * len(train_dataloader)
        # Epoch-based warmup gate (e.g. 2 epochs)
        warmup_steps = warmup_epochs * len(train_dataloader)
        print(f"🛡️ Adversarial Shield: {warmup_epochs}-epoch Lambda Warmup enabled ({warmup_steps} steps gate)")
        
        best_val_f1 = 0.0
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_dataloader, optimizer, epoch, warmup_steps, grl_lambda_max)
            
            # Validate
            val_metrics = self.validate(val_dataloader, epoch)
            
            # LR scheduling
            scheduler.step()
            
            # Logging
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Emotion Acc: {train_metrics['emotion_accuracy']:.4f}")
            print(f"🛡️  DAT   - Lambda: {train_metrics['last_lambda']:.4f}, "
                  f"Disc Acc: {train_metrics['disc_accuracy']:.4f}, "
                  f"Grad Norm: {train_metrics['grad_norm']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Emotion Acc: {val_metrics['emotion_accuracy']:.4f}, "
                  f"Intensity MSE: {val_metrics['intensity_mse']:.4f}")
            
            # Store history
            self.training_history["loss"].append(val_metrics["loss"])
            self.training_history["emotion_acc"].append(val_metrics["emotion_accuracy"])
            self.training_history["sarcasm_acc"].append(val_metrics["sarcasm_accuracy"])
            self.training_history["intensity_mse"].append(val_metrics["intensity_mse"])
            self.training_history["disc_acc"].append(train_metrics["disc_accuracy"])
            self.training_history["grad_norm"].append(train_metrics["grad_norm"])
            
            # Save best model (based on F1 Macro to avoid bias)
            if val_metrics["emotion_f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["emotion_f1_macro"]
                self.save_checkpoint(epoch, val_metrics)
                print(f"✓ Saved best model (Emotion F1-Macro: {best_val_f1:.4f})")
                
                # Save emotion centroids for explainability
                self._save_emotion_centroids(val_metrics["embeddings_by_emotion"])
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "training_history": self.training_history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pt"))
        print(f"  ✓ Checkpoint saved to {self.save_dir}/best_model.pt")
    
    def _save_emotion_centroids(self, embeddings_by_emotion: Dict):
        """Save emotion centroids for explainability engine."""
        centroids = {}
        for emotion_idx, embeddings_list in embeddings_by_emotion.items():
            if len(embeddings_list) > 0:
                centroid = np.mean(embeddings_list, axis=0)
                centroids[emotion_idx] = centroid
        
        with open(os.path.join(self.save_dir, "emotion_centroids.pkl"), "wb") as f:
            pickle.dump(centroids, f)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint.get("training_history", self.training_history)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
