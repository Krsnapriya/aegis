"""
Tests for the PluTchikMultiTaskModel.
"""

import pytest
import torch
from plutchik_erc_dashboard.models.multitask_emotion_model import PluTchikMultiTaskModel, MultiTaskLoss

@pytest.fixture
def model():
    """Returns a PluTchikMultiTaskModel instance."""
    return PluTchikMultiTaskModel(num_emotions=32, pretrained_model="prajjwal1/bert-tiny")

@pytest.fixture
def loss_fn():
    """Returns a MultiTaskLoss instance."""
    return MultiTaskLoss()

def test_model_initialization(model):
    """Tests if the model can be initialized."""
    assert model is not None
    assert isinstance(model, PluTchikMultiTaskModel)

def test_model_forward_pass(model):
    """Tests the forward pass of the model."""
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids, attention_mask)
    
    assert "emotion_logits" in outputs
    assert "sarcasm_logits" in outputs
    assert "intensity" in outputs
    
    assert outputs["emotion_logits"].shape == (batch_size, 32)
    assert outputs["sarcasm_logits"].shape == (batch_size, 2)
    assert outputs["intensity"].shape == (batch_size, 1)

def test_loss_calculation(loss_fn):
    """Tests the MultiTaskLoss calculation."""
    batch_size = 4
    num_emotions = 32
    
    predictions = {
        "emotion_logits": torch.randn(batch_size, num_emotions),
        "sarcasm_logits": torch.randn(batch_size, 2),
        "intensity": torch.rand(batch_size, 1),
        "scenario_logits": torch.randn(batch_size, 2),
    }
    
    targets = {
        "emotion": torch.randint(0, num_emotions, (batch_size,)),
        "sarcasm": torch.randint(0, 2, (batch_size,)),
        "intensity": torch.rand(batch_size, 1),
        "scenario": torch.randint(0, 2, (batch_size,)),
    }
    
    total_loss, loss_breakdown = loss_fn(predictions, targets)
    
    assert isinstance(total_loss, torch.Tensor)
    assert "emotion_loss" in loss_breakdown
    assert "sarcasm_loss" in loss_breakdown
    assert "intensity_loss" in loss_breakdown
    assert "adv_loss" in loss_breakdown
    assert total_loss.item() >= 0
