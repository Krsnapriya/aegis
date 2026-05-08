"""
Multi-Task Emotion Recognition Model with RoBERTa backbone.
Predicts: Emotion (32-class), Sarcasm (Binary), Intensity (Regression 0-1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class PluTchikMultiTaskModel(nn.Module):
    """
    Multi-task emotion recognition model with three prediction heads:
    1. Emotion Classification (32 classes)
    2. Sarcasm Detection (Binary)
    3. Intensity Regression (0-1)
    """
    
    def __init__(self, num_emotions=32, hidden_dim=768, dropout=0.2, pretrained_model="roberta-base"):
        super(PluTchikMultiTaskModel, self).__init__()
        
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        
        # RoBERTa backbone
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        
        # Shared dense layers
        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # ========== EMOTION HEAD ==========
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
        
        # ========== SARCASM HEAD (Binary) ==========
        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # ========== INTENSITY HEAD (Regression) ==========
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # ========== ADVERSARIAL DOMAIN DISCRIMINATOR ==========
        # Binary: Formal/Professional vs Social/Personal
        self.scenario_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary Domain Discrimination
        )
        
        # Store last hidden states for explainability
        self.last_hidden_state = None
        self.token_ids = None
        
        # ========== STAGE 3: DISSONANCE HEAD (Dual-Encoder) ==========
        # Separate context pooler — shares RoBERTa weights, separate CLS pathway
        self.context_pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # Dissonance head: [text_rep (768) || context_rep (384)] → dissonance score [0,1]
        self.dissonance_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, alpha=0.0,
                context_input_ids=None, context_attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        
        Returns:
            Dict with emotion_logits, sarcasm_logits, intensity_predictions, last_hidden_state
        """
        # Store token IDs for explainability
        self.token_ids = input_ids
        
        # RoBERTa forward pass
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # CLS token representation [batch_size, hidden_dim]
        cls_hidden_state = roberta_output.last_hidden_state[:, 0, :]
        
        # Store for explainability (full sequence)
        self.last_hidden_state = roberta_output.last_hidden_state
        
        # Shared dense layer
        shared_rep = self.shared_dense(cls_hidden_state)
        
        # Three heads
        emotion_logits = self.emotion_head(shared_rep)
        sarcasm_logits = self.sarcasm_head(shared_rep)
        intensity_pred = torch.sigmoid(self.intensity_head(shared_rep))  # [0, 1] range
        
        # Adversarial Head: Gradient Reversal Layer (GRL)
        # alpha is the penalty weight (warmup is handled in the training loop)
        reverse_feature = GradReverse.apply(shared_rep, alpha)
        scenario_logits = self.scenario_discriminator(reverse_feature)
        
        # ── Dissonance path (only active for contrastive batches) ───────────────
        dissonance_score = None
        if context_input_ids is not None:
            ctx_output = self.roberta(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                return_dict=True
            )
            ctx_cls = ctx_output.last_hidden_state[:, 0, :]
            ctx_rep = self.context_pooler(ctx_cls)
            joint_rep = torch.cat([shared_rep, ctx_rep], dim=-1)
            dissonance_score = self.dissonance_head(joint_rep)

        return {
            "emotion_logits": emotion_logits,
            "sarcasm_logits": sarcasm_logits,
            "intensity": intensity_pred,
            "scenario_logits": scenario_logits,
            "last_hidden_state": self.last_hidden_state,
            "cls_embedding": cls_hidden_state,
            "shared_representation": shared_rep,
            "dissonance_score": dissonance_score
        }
    
    def get_embeddings_for_explainability(self):
        """Return stored embeddings for visualization."""
        return {
            "token_ids": self.token_ids,
            "last_hidden_state": self.last_hidden_state
        }
    
    def get_embeddings(self, text, tokenizer, device="cpu"):
        """
        Extract RoBERTa embeddings for the given text.
        Returns: A list of floats (CLS token embedding)
        """
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = self.roberta(**inputs)
            # Use CLS token (index 0)
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy().tolist()
            return cls_embedding


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining:
    - Cross-Entropy for emotion classification
    - Cross-Entropy for sarcasm detection
    - MSE for intensity regression
    """
    
    def __init__(self, emotion_weight=1.0, sarcasm_weight=0.7, intensity_weight=0.5, 
                 adv_weight=0.3, dissonance_weight=1.0, iaa_weighting=False):
        super(MultiTaskLoss, self).__init__()
        self.emotion_weight = emotion_weight
        self.sarcasm_weight = sarcasm_weight
        self.intensity_weight = intensity_weight
        self.adv_weight = adv_weight
        self.dissonance_weight = dissonance_weight
        self.iaa_weighting = iaa_weighting
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, sample_weights=None):
        """
        Calculate weighted multi-task loss.
        
        Args:
            predictions: Dict from model forward pass
            targets: Dict with keys 'emotion', 'sarcasm', 'intensity'
            sample_weights: Inter-annotator agreement weights [batch_size]
        
        Returns:
            Total loss (scalar)
        """
        batch_size = predictions["emotion_logits"].shape[0]
        
        # Emotion loss (32-way classification)
        emotion_loss = self.ce_loss(
            predictions["emotion_logits"],
            targets["emotion"]
        )
        
        # Sarcasm loss (binary classification)
        sarcasm_loss = self.ce_loss(
            predictions["sarcasm_logits"],
            targets["sarcasm"]
        )
        
        # Intensity loss (regression)
        intensity_loss = self.mse_loss(
            predictions["intensity"],
            targets["intensity"]
        )
        
        # Apply IAA weighting if provided
        if sample_weights is not None:
            emotion_loss = emotion_loss * sample_weights
            sarcasm_loss = sarcasm_loss * sample_weights
            intensity_loss = intensity_loss.squeeze() * sample_weights
        
        # Scenario loss (adversarial)
        adv_loss = self.ce_loss(
            predictions["scenario_logits"],
            targets["scenario"]
        )
        
        # Total weighted loss
        total_loss = (
            self.emotion_weight * emotion_loss.mean() +
            self.sarcasm_weight * sarcasm_loss.mean() +
            self.intensity_weight * intensity_loss.mean() +
            self.adv_weight * adv_loss.mean() 
        )

        dissonance_loss_val = 0.0
        if "dissonance" in targets and targets["dissonance"] is not None:
            if predictions.get("dissonance_score") is not None and targets["dissonance"].sum() > 0:
                dissonance_loss = F.binary_cross_entropy(
                    predictions["dissonance_score"].squeeze(),
                    targets["dissonance"].float()
                )
                total_loss += self.dissonance_weight * dissonance_loss
                dissonance_loss_val = dissonance_loss.item()
        
        return total_loss, {
            "emotion_loss": emotion_loss.mean().item(),
            "sarcasm_loss": sarcasm_loss.mean().item(),
            "intensity_loss": intensity_loss.mean().item(),
            "adv_loss": adv_loss.mean().item(),
            "dissonance_loss": dissonance_loss_val
        }
