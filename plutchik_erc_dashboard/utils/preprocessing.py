"""
Deep ERC Preprocessing Pipeline with Metadata and Context Windows.
Handles dialogue context, metadata augmentation, and dataset preparation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from typing import Dict, List, Tuple, Optional


class ERCPreprocessor:
    """
    Preprocesses ERC dialogue data with scenario, topic, and context window.
    """
    
    def __init__(self, plutchik_dict: Dict, tokenizer_name: str = "roberta-base"):
        """
        Initialize preprocessor.
        
        Args:
            plutchik_dict: Dictionary mapping emotion names to metadata
            tokenizer_name: HuggingFace tokenizer identifier
        """
        self.plutchik = plutchik_dict
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(plutchik_dict.keys()))}
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        
        self.scenarios = ["workplace", "friendship", "family", "romance", "support", "academic", 
                          "conflict", "casual", "social", "travel", "technology", "creative", 
                          "wellbeing", "community"]
        
        # Binary Domain Mapping for Adversarial Hardening
        # Based on data-derived sarcasm rates (median split)
        # Group 0: High-Sarcasm scenarios, Group 1: Low-Sarcasm scenarios
        # Audit: workplace(47%), social(32%), conflict(24%), casual(22%), friendship(13%), romance(9%)
        self.high_sarcasm_scenarios = {
            "workplace", "social", "conflict", "casual", "friendship", "romance"
        }
        self.scenario_to_idx = {s: (0 if s in self.high_sarcasm_scenarios else 1) for s in self.scenarios}
    
    def augment_with_metadata(self, text: str, scenario: str, topic: str) -> str:
        """
        Prepend scenario and topic metadata to text.
        
        Format: [SCENARIO] workplace [/SCENARIO] [TOPIC] termination [/TOPIC] <original_text>
        
        Args:
            text: Original utterance
            scenario: Workplace, friendship, family, etc.
            topic: conversation topic
        
        Returns:
            Augmented text with metadata
        """
        augmented = f"[SCENARIO] {scenario} [/SCENARIO] [TOPIC] {topic} [/TOPIC] {text}"
        return augmented
    
    def get_context_window(self, dialogues: List[Tuple], current_idx: int, 
                          window_size: int = 2) -> str:
        """
        Retrieve previous N turns from the dialogue to capture emotional shift.
        
        Args:
            dialogues: List of (speaker, text, emotion, sarcasm_flag, emotion_cause) tuples
            current_idx: Index of current utterance
            window_size: Number of previous turns to include
        
        Returns:
            Concatenated context string
        """
        context_turns = []
        
        # Include previous turns (up to window_size)
        start_idx = max(0, current_idx - window_size)
        for idx in range(start_idx, current_idx):
            speaker, text, _, _, _ = dialogues[idx]
            # Abbreviate speaker for context
            speaker_abbr = speaker.split('_')[0][:3]
            context_turns.append(f"{speaker_abbr}: {text}")
        
        context_window = " | ".join(context_turns) if context_turns else "[NO_CONTEXT]"
        return context_window
    
    def prepare_sample(self, speaker: str, text: str, emotion: str, 
                      sarcasm_flag: bool, emotion_cause: Optional[str],
                      scenario: str, topic: str, dialogues: List[Tuple],
                      current_idx: int, iaa_score: float = 0.75,
                      row_data: Dict = None) -> Dict:
        """
        Prepare a single training sample with all augmentations.
        
        Args:
            speaker: Speaker name
            text: Utterance text
            emotion: Target emotion label
            sarcasm_flag: Whether utterance contains sarcasm
            emotion_cause: Explanation of emotion trigger
            scenario: Dialogue scenario
            topic: Dialogue topic
            dialogues: Full dialogue list (for context)
            current_idx: Current utterance index
            iaa_score: Inter-annotator agreement score for weighting
        
        Returns:
            Dict with processed sample
        """
        # Augment with scenario and topic
        augmented_text = self.augment_with_metadata(text, scenario, topic)
        
        # Get context window
        context = self.get_context_window(dialogues, current_idx, window_size=2)
        
        # Combine augmented text with context
        full_input = f"[CONTEXT] {context} [/CONTEXT] [CURRENT] {augmented_text} [/CURRENT]"
        
        # Tokenize
        encoding = self.tokenizer(
            full_input,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get emotion index
        emotion_idx = self.emotion_to_idx[emotion]
        
        # Sarcasm as binary (0 or 1)
        sarcasm_idx = int(sarcasm_flag)
        
        # Intensity: map primary emotions to 0.5, intense to 1.0, mild to 0.25
        ring = self.plutchik[emotion].get("ring", "primary")
        if ring == "intense":
            intensity = 1.0
        elif ring == "primary":
            intensity = 0.5
        elif ring == "mild":
            intensity = 0.25
        elif ring == "dyadic":
            intensity = 0.6
        else:
            intensity = 0.5
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "emotion_label": emotion_idx,
            "sarcasm_label": sarcasm_idx,
            "intensity_label": intensity,
            "iaa_weight": iaa_score,
            "emotion_name": emotion,
            "speaker": speaker,
            "scenario": scenario,
            "scenario_label": self.scenario_to_idx.get(scenario, 0),
            "topic": topic,
            "emotion_cause": emotion_cause if pd.notna(emotion_cause) else "Not specified",
            "full_text": full_input,
            "split": row_data.get("split", "train") if isinstance(row_data, dict) else "train"
        }


class PlutchikERCDataset(Dataset):
    """
    PyTorch Dataset for Plutchik ERC data with multi-task labels.
    """
    
    def __init__(self, samples: List[Dict]):
        """
        Args:
            samples: List of preprocessed samples from ERCPreprocessor
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        batch = {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "emotion_label": torch.tensor(sample["emotion_label"], dtype=torch.long),
            "sarcasm_label": torch.tensor(sample["sarcasm_label"], dtype=torch.long),
            "intensity_label": torch.tensor(sample["intensity_label"], dtype=torch.float32).unsqueeze(0),
            "iaa_weight": torch.tensor(sample["iaa_weight"], dtype=torch.float32),
            "scenario_label": torch.tensor(sample["scenario_label"], dtype=torch.long),
        }

        # Optional Dissonance Head Fields
        if "context_input_ids" in sample:
            batch["context_input_ids"] = sample["context_input_ids"]
            batch["context_attention_mask"] = sample["context_attention_mask"]
            batch["dissonance_label"] = torch.tensor(sample["dissonance_label"], dtype=torch.float32)
        
        return batch


def build_dataset_from_dialogues(dialogues_list: List[Dict], plutchik_dict: Dict,
                                  tokenizer_name: str = "roberta-base") -> PlutchikERCDataset:
    """
    Build complete dataset from dialogue list (DIALOGUES constant format).
    """
    preprocessor = ERCPreprocessor(plutchik_dict, tokenizer_name)
    samples = []
    
    for dialogue_dict in dialogues_list:
        scenario = dialogue_dict["scenario"]
        topic = dialogue_dict["topic"]
        utterances = dialogue_dict["utterances"]
        
        for idx, utterance in enumerate(utterances):
            speaker, text, emotion, sarcasm_flag, emotion_cause = utterance
            
            sample = preprocessor.prepare_sample(
                speaker=speaker,
                text=text,
                emotion=emotion,
                sarcasm_flag=sarcasm_flag,
                emotion_cause=emotion_cause,
                scenario=scenario,
                topic=topic,
                dialogues=utterances,
                current_idx=idx,
                iaa_score=0.80
            )
            samples.append(sample)
    
    return PlutchikERCDataset(samples)


def build_dataset_from_csv(csv_path: str, plutchik_dict: Dict,
                            tokenizer_name: str = "roberta-base", split: str = None) -> PlutchikERCDataset:
    """
    Build dataset by loading CSV and grouping by dialogue_id for context.
    Optionally filters by split (train/val/test).
    """
    df = pd.read_csv(csv_path)
    if split:
        df = df[df["split"] == split]
    
    preprocessor = ERCPreprocessor(plutchik_dict, tokenizer_name)
    samples = []
    
    # Group by dialogue_id to preserve context
    dialogues = df.groupby("dialogue_id")
    
    for _, group in dialogues:
        group = group.sort_values("turn_id")
        utterances_list = []
        for _, row in group.iterrows():
            utterances_list.append((
                row["speaker"], 
                row["text"], 
                row["emotion"], 
                row["sarcasm_flag"], 
                row["emotion_cause"]
            ))
            
        for idx, row in group.reset_index().iterrows():
            # Pass full row as dict to prepare_sample for extra metadata (like split)
            row_dict = row.to_dict()
            sample = preprocessor.prepare_sample(
                speaker=row["speaker"],
                text=row["text"],
                emotion=row["emotion"],
                sarcasm_flag=row["sarcasm_flag"],
                emotion_cause=row["emotion_cause"],
                scenario=row["scenario"],
                topic=row["topic"],
                dialogues=utterances_list,
                current_idx=idx,
                iaa_score=row["inter_annotator_agreement"],
                row_data=row_dict
            )
    return PlutchikERCDataset(samples)


def load_contrastive_pairs(jsonl_path: str, plutchik_dict: Dict, tokenizer_name: str = "roberta-base") -> List[Dict]:
    """
    Load human-verified contrastive pairs for dissonance head training.
    """
    import json
    import os
    preprocessor = ERCPreprocessor(plutchik_dict, tokenizer_name)
    samples = []
    
    if not os.path.exists(jsonl_path):
        return []
        
    with open(jsonl_path, 'r') as f:
        for line in f:
            pair = json.loads(line)
            # pair_verifier.py appends both 'pair' and 'twin' to the file.
            dissonance_score = pair.get('dissonance_score', 1.0)
            is_dissonant = dissonance_score > 0.5
            
            context = pair['original_context'] if is_dissonant else pair['twin_context']
            emotion = pair['original_emotion'] if is_dissonant else pair['twin_emotion']
            
            # Prepare sample
            augmented_text = preprocessor.augment_with_metadata(pair['text'], pair['scenario'], "general")
            full_input = f"[CONTEXT] {context} [/CONTEXT] [CURRENT] {augmented_text} [/CURRENT]"
            
            encoding = preprocessor.tokenizer(
                full_input, max_length=256, padding='max_length', truncation=True, return_tensors='pt'
            )
            
            # For the Dual-Encoder dissonance head, we also need the context *alone*
            ctx_encoding = preprocessor.tokenizer(
                context, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
            )
            
            samples.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "context_input_ids": ctx_encoding["input_ids"].squeeze(),
                "context_attention_mask": ctx_encoding["attention_mask"].squeeze(),
                "emotion_label": preprocessor.emotion_to_idx.get(emotion, 0),
                "sarcasm_label": 1 if is_dissonant else 0,
                "intensity_label": 0.8 if is_dissonant else 0.4,
                "dissonance_label": dissonance_score,
                "iaa_weight": 2.0, 
                "scenario_label": preprocessor.scenario_to_idx.get(pair['scenario'], 0),
                "split": "train"
            })
    return samples
