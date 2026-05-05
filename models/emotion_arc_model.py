"""
Plutchik ERC v2.1 — Emotion Arc Model
Dialogue-level sequence model that predicts emotion trajectories.

Architecture:
    Input:  Sequence of CLS embeddings from the utterance-level encoder
            [num_turns, hidden_dim] per dialogue
    Core:   Bidirectional GRU with attention pooling
    Output: Per-turn emotion prediction + dialogue-level arc features

This model sits ON TOP of the existing PluTchikMultiTaskModel.
It does not replace it — it consumes its CLS embeddings as input.

Key Capabilities:
    1. Predict the next emotion in a conversation (autoregressive)
    2. Classify the overall emotional trajectory (escalation, de-escalation, stable, volatile)
    3. Detect emotional turning points in a dialogue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============== ARC TRAJECTORY LABELS ==============
ARC_LABELS = {
    0: "escalation",      # Emotion intensity increases over turns
    1: "de-escalation",   # Emotion intensity decreases over turns
    2: "stable",          # Emotion remains consistent
    3: "volatile",        # Emotion oscillates unpredictably
}


class TemporalAttention(nn.Module):
    """
    Learned attention over GRU hidden states to weight
    which turns matter most for the arc classification.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, gru_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_outputs: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] — 1 for real turns, 0 for padding

        Returns:
            context: [batch, hidden_dim] — weighted sum
            weights: [batch, seq_len] — attention weights
        """
        scores = self.attention(gru_outputs).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), gru_outputs).squeeze(1)  # [batch, hidden_dim]

        return context, weights


class EmotionArcModel(nn.Module):
    """
    Dialogue-level emotion trajectory model.

    Consumes a sequence of CLS embeddings from the utterance encoder
    and predicts:
        1. Per-turn emotion (next-turn prediction)
        2. Dialogue arc type (escalation / de-escalation / stable / volatile)
        3. Turning point scores (which turn caused the biggest shift)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 num_emotions: int = 32, num_arc_classes: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions

        # Input projection (from RoBERTa CLS dim to GRU dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Temporal attention for arc classification
        self.attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional

        # ========== HEAD 1: Per-Turn Emotion Prediction ==========
        # Uses the GRU hidden state at each timestep
        self.turn_emotion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )

        # ========== HEAD 2: Arc Classification ==========
        # Uses attention-pooled representation of the full dialogue
        self.arc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_arc_classes)
        )

        # ========== HEAD 3: Turning Point Detection ==========
        # Per-turn binary score: is this turn a turning point?
        self.turning_point_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, cls_embeddings: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            cls_embeddings: [batch, max_turns, input_dim]
                Sequence of CLS embeddings from the utterance encoder.
            lengths: [batch] — actual number of turns per dialogue (for packing)
            mask: [batch, max_turns] — 1 for real turns, 0 for padding

        Returns:
            Dict with:
                turn_emotion_logits: [batch, max_turns, num_emotions]
                arc_logits: [batch, num_arc_classes]
                turning_point_scores: [batch, max_turns]
                attention_weights: [batch, max_turns]
        """
        batch_size, max_turns, _ = cls_embeddings.shape

        # Project input
        projected = self.input_proj(cls_embeddings)  # [batch, max_turns, hidden_dim]

        # Pack sequences if lengths provided (for efficiency)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=max_turns
            )
        else:
            gru_out, _ = self.gru(projected)  # [batch, max_turns, hidden_dim*2]

        # Generate mask from lengths if not provided
        if mask is None and lengths is not None:
            mask = torch.arange(max_turns, device=cls_embeddings.device).unsqueeze(0) < lengths.to(cls_embeddings.device).unsqueeze(1)
            mask = mask.float()

        # Head 1: Per-turn emotion prediction
        turn_emotion_logits = self.turn_emotion_head(gru_out)  # [batch, max_turns, num_emotions]

        # Head 2: Arc classification (attention-pooled)
        arc_context, attn_weights = self.attention(gru_out, mask)
        arc_logits = self.arc_head(arc_context)  # [batch, num_arc_classes]

        # Head 3: Turning point detection
        turning_point_scores = self.turning_point_head(gru_out).squeeze(-1)  # [batch, max_turns]

        return {
            "turn_emotion_logits": turn_emotion_logits,
            "arc_logits": arc_logits,
            "turning_point_scores": turning_point_scores,
            "attention_weights": attn_weights,
        }


class EmotionArcLoss(nn.Module):
    """
    Combined loss for the Emotion Arc model.

    Components:
        1. Per-turn emotion CE (teacher-forced)
        2. Arc classification CE
        3. Turning point BCE
    """

    def __init__(self, emotion_weight: float = 1.0,
                 arc_weight: float = 0.5,
                 turning_point_weight: float = 0.3):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.arc_weight = arc_weight
        self.turning_point_weight = turning_point_weight

        self.emotion_ce = nn.CrossEntropyLoss(reduction='none')
        self.arc_ce = nn.CrossEntropyLoss()
        self.tp_bce = nn.BCELoss(reduction='none')

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Output from EmotionArcModel.forward()
            targets:
                turn_emotions: [batch, max_turns] — emotion labels per turn
                arc_label: [batch] — arc type label
                turning_points: [batch, max_turns] — binary turning point labels
            mask: [batch, max_turns] — 1 for real turns

        Returns:
            total_loss, loss_breakdown
        """
        batch_size, max_turns, num_emotions = predictions["turn_emotion_logits"].shape

        # 1. Per-turn emotion loss (masked)
        turn_logits = predictions["turn_emotion_logits"].reshape(-1, num_emotions)
        turn_targets = targets["turn_emotions"].reshape(-1)
        turn_loss = self.emotion_ce(turn_logits, turn_targets)
        turn_loss = turn_loss.reshape(batch_size, max_turns)

        if mask is not None:
            turn_loss = (turn_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            turn_loss = turn_loss.mean()

        # 2. Arc classification loss
        arc_loss = self.arc_ce(predictions["arc_logits"], targets["arc_label"])

        # 3. Turning point loss (masked)
        tp_loss = self.tp_bce(
            predictions["turning_point_scores"],
            targets["turning_points"].float()
        )
        if mask is not None:
            tp_loss = (tp_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            tp_loss = tp_loss.mean()

        total = (
            self.emotion_weight * turn_loss +
            self.arc_weight * arc_loss +
            self.turning_point_weight * tp_loss
        )

        return total, {
            "turn_emotion_loss": turn_loss.item(),
            "arc_loss": arc_loss.item(),
            "turning_point_loss": tp_loss.item(),
        }


# ============== ARC UTILITIES ==============

def compute_arc_label(emotion_indices: List[int], plutchik_dict: dict) -> int:
    """
    Heuristic arc labeler: computes the trajectory type from a sequence
    of emotion indices.

    Uses the ring intensity mapping to convert emotions to a scalar
    intensity sequence, then classifies the trajectory shape.

    Args:
        emotion_indices: List of emotion indices (sorted PLUTCHIK keys order)
        plutchik_dict: The PLUTCHIK dictionary

    Returns:
        Arc label (0=escalation, 1=de-escalation, 2=stable, 3=volatile)
    """
    from utils.constants import RING_INTENSITY

    emotion_names = sorted(plutchik_dict.keys())
    intensities = []

    for idx in emotion_indices:
        emotion = emotion_names[idx]
        ring = plutchik_dict[emotion]["ring"]
        intensities.append(RING_INTENSITY.get(ring, 0.5))

    if len(intensities) < 2:
        return 2  # stable (single turn)

    # Compute deltas
    deltas = [intensities[i+1] - intensities[i] for i in range(len(intensities) - 1)]
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    # Classification heuristic
    if std_delta > 0.25:
        return 3  # volatile — high variance in direction
    elif mean_delta > 0.1:
        return 0  # escalation — trending upward
    elif mean_delta < -0.1:
        return 1  # de-escalation — trending downward
    else:
        return 2  # stable — minimal change


def detect_turning_points(emotion_indices: List[int], plutchik_dict: dict,
                          threshold: float = 0.3) -> List[int]:
    """
    Detect turns where the emotional register shifts significantly.

    A turning point is any turn where the intensity delta from the
    previous turn exceeds the threshold.

    Args:
        emotion_indices: List of emotion indices
        plutchik_dict: The PLUTCHIK dictionary
        threshold: Minimum intensity delta to count as a turning point

    Returns:
        List of turn indices that are turning points
    """
    from utils.constants import RING_INTENSITY

    emotion_names = sorted(plutchik_dict.keys())
    intensities = []

    for idx in emotion_indices:
        emotion = emotion_names[idx]
        ring = plutchik_dict[emotion]["ring"]
        intensities.append(RING_INTENSITY.get(ring, 0.5))

    turning_points = []
    for i in range(1, len(intensities)):
        delta = abs(intensities[i] - intensities[i-1])
        if delta >= threshold:
            turning_points.append(i)

    return turning_points


def prepare_arc_batch(dialogues: List[Dict], utterance_model,
                      tokenizer, device: str = "cpu",
                      max_turns: int = 20) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of dialogues for the EmotionArcModel.

    Takes raw dialogue dicts, runs each utterance through the utterance-level
    encoder, and collects CLS embeddings into padded sequences.

    Args:
        dialogues: List of dialogue dicts with 'utterances' key
        utterance_model: Trained PluTchikMultiTaskModel
        tokenizer: RoBERTa tokenizer
        device: Target device
        max_turns: Maximum turns to consider per dialogue

    Returns:
        Dict with cls_embeddings, lengths, mask, turn_emotions, arc_labels, turning_points
    """
    from utils.constants import PLUTCHIK

    emotion_names = sorted(PLUTCHIK.keys())
    emotion_to_idx = {e: i for i, e in enumerate(emotion_names)}

    all_embeddings = []
    all_lengths = []
    all_turn_emotions = []
    all_arc_labels = []
    all_turning_points = []

    utterance_model.eval()

    # Pre-collect all utterances across all dialogues for batch inference
    all_utts_texts = []
    dialogue_indices = []
    for d_idx, dialogue in enumerate(dialogues):
        utts = dialogue["utterances"][:max_turns]
        all_utts_texts.extend([u[1] for u in utts]) # text is index 1
        dialogue_indices.extend([d_idx] * len(utts))
    
    if not all_utts_texts:
        return {}

    # Batch Inference on all utterances
    all_cls_embeddings = []
    batch_size = 16 # Safe batch size for RoBERTa
    for i in range(0, len(all_utts_texts), batch_size):
        batch_texts = all_utts_texts[i:i+batch_size]
        encodings = tokenizer(
            batch_texts, max_length=256, padding='max_length',
            truncation=True, return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = utterance_model(encodings["input_ids"], encodings["attention_mask"])
            all_cls_embeddings.append(outputs["cls_embedding"].cpu())
    
    all_cls_embeddings = torch.cat(all_cls_embeddings, dim=0)

    # Reconstruct dialogues
    curr_idx = 0
    for d_idx, dialogue in enumerate(dialogues):
        utterances = dialogue["utterances"][:max_turns]
        num_turns = len(utterances)
        
        turn_embeddings = all_cls_embeddings[curr_idx : curr_idx + num_turns]
        turn_emotion_indices = [emotion_to_idx.get(u[2], 0) for u in utterances] # emotion is index 2
        curr_idx += num_turns

        # Pad to max_turns
        emb_dim = turn_embeddings[0].shape[0]
        padded_embeddings = torch.zeros(max_turns, emb_dim)
        padded_emotions = torch.zeros(max_turns, dtype=torch.long)
        padded_tp = torch.zeros(max_turns)

        for i, emb in enumerate(turn_embeddings):
            padded_embeddings[i] = emb
            padded_emotions[i] = turn_emotion_indices[i]

        # Compute arc label and turning points
        arc_label = compute_arc_label(turn_emotion_indices, PLUTCHIK)
        tp_indices = detect_turning_points(turn_emotion_indices, PLUTCHIK)
        for tp_idx in tp_indices:
            if tp_idx < max_turns:
                padded_tp[tp_idx] = 1.0

        all_embeddings.append(padded_embeddings)
        all_lengths.append(num_turns)
        all_turn_emotions.append(padded_emotions)
        all_arc_labels.append(arc_label)
        all_turning_points.append(padded_tp)

    # Stack into batch tensors
    cls_embeddings = torch.stack(all_embeddings).to(device)
    lengths = torch.tensor(all_lengths, dtype=torch.long)
    mask = torch.arange(max_turns).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.float().to(device)

    return {
        "cls_embeddings": cls_embeddings,
        "lengths": lengths,
        "mask": mask,
        "turn_emotions": torch.stack(all_turn_emotions).to(device),
        "arc_labels": torch.tensor(all_arc_labels, dtype=torch.long).to(device),
        "turning_points": torch.stack(all_turning_points).to(device),
    }
