"""
Tests for the ERC Preprocessing Pipeline.
"""

import pytest
import torch
from utils.preprocessing import ERCPreprocessor, PlutchikERCDataset
from utils.constants import PLUTCHIK

@pytest.fixture
def preprocessor():
    """Returns an ERCPreprocessor instance."""
    return ERCPreprocessor(plutchik_dict=PLUTCHIK)

def test_augment_with_metadata(preprocessor):
    """Tests the metadata augmentation."""
    text = "This is a test."
    scenario = "workplace"
    topic = "release"
    augmented_text = preprocessor.augment_with_metadata(text, scenario, topic)
    expected = "[SCENARIO] workplace [/SCENARIO] [TOPIC] release [/TOPIC] This is a test."
    assert augmented_text == expected

def test_get_context_window(preprocessor):
    """Tests the context window retrieval."""
    dialogues = [
        ("USER", "Hello.", "serenity", False, None),
        ("AGENT", "Hi, how can I help?", "interest", False, None),
        ("USER", "I have a problem.", "sadness", False, None),
    ]
    context = preprocessor.get_context_window(dialogues, current_idx=2, window_size=2)
    expected = "USE: Hello. | AGE: Hi, how can I help?"
    assert context == expected

def test_prepare_sample(preprocessor):
    """Tests the full sample preparation."""
    dialogues = [
        ("USER", "I am so happy!", "joy", False, None),
    ]
    sample = preprocessor.prepare_sample(
        speaker="USER",
        text="I am so happy!",
        emotion="joy",
        sarcasm_flag=False,
        emotion_cause=None,
        scenario="friendship",
        topic="party",
        dialogues=dialogues,
        current_idx=0,
        iaa_score=0.9
    )
    
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert sample["emotion_label"] == preprocessor.emotion_to_idx["joy"]
    assert sample["sarcasm_label"] == 0
    assert sample["intensity_label"] == 0.5  # Joy is a primary emotion
    assert "full_text" in sample
    assert "[CONTEXT] [NO_CONTEXT] [/CONTEXT]" in sample["full_text"]
    assert "[SCENARIO] friendship [/SCENARIO]" in sample["full_text"]

def test_dataset_creation(preprocessor):
    """Tests the PlutchikERCDataset class."""
    dialogues = [
        ("USER", "First message", "interest", False, None),
        ("AGENT", "Second message", "boredom", False, None),
    ]
    samples = [
        preprocessor.prepare_sample(
            "USER", "First message", "interest", False, None, "casual", "greeting", dialogues, 0
        ),
        preprocessor.prepare_sample(
            "AGENT", "Second message", "boredom", False, None, "casual", "greeting", dialogues, 1
        )
    ]
    dataset = PlutchikERCDataset(samples)
    assert len(dataset) == 2
    
    item = dataset[0]
    assert "input_ids" in item
    assert "emotion_label" in item
    assert isinstance(item["input_ids"], torch.Tensor)
