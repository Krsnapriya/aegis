import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

EMOTION_CLASSES = [
    'ecstasy', 'admiration', 'terror', 'amazement', 'grief', 'loathing',
    'rage', 'vigilance', 'joy', 'trust', 'fear', 'surprise', 'sadness',
    'disgust', 'anger', 'anticipation', 'serenity', 'acceptance',
    'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance',
    'interest', 'optimism', 'love', 'submission', 'awe', 'disapproval',
    'remorse', 'contempt', 'aggressiveness'
]

INTENSITY_LABELS = ['mild', 'primary', 'intense']

class PluTchikMultiTaskModel(nn.Module):
    def __init__(self, num_emotions=32, dropout=0.3):
        super().__init__()
        # Use tiny config: 2 layers, 128 hidden, 2 heads = ~5MB model
        self.config = BertConfig(
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=128
        )
        self.bert = BertModel(self.config)
        
        hidden_size = 128
        
        # Emotion classification head (32 classes)
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_emotions)
        )
        
        # Sarcasm detection head (binary)
        self.sarcasm_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Intensity regression head (3-class softmax treated as ordinal)
        self.intensity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None, 
                sarcasm_labels=None, intensity_labels=None):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Emotion logits
        emotion_logits = self.emotion_classifier(pooled_output)
        
        # Sarcasm logits
        sarcasm_logits = self.sarcasm_head(pooled_output).squeeze(-1)
        
        # Intensity logits
        intensity_logits = self.intensity_head(pooled_output)
        
        loss_dict = {}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_dict['emotion_loss'] = loss_fct(emotion_logits, labels)
        
        if sarcasm_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss_dict['sarcasm_loss'] = loss_fct(sarcasm_logits, sarcasm_labels.float())
        
        if intensity_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_dict['intensity_loss'] = loss_fct(intensity_logits, intensity_labels)
        
        total_loss = sum(loss_dict.values()) if loss_dict else None
        
        return {
            'loss': total_loss,
            'losses': loss_dict,
            'emotion_logits': emotion_logits,
            'sarcasm_logits': sarcasm_logits,
            'intensity_logits': intensity_logits,
            'pooled_output': pooled_output
        }
