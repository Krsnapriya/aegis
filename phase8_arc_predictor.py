"""
Phase 8 Extension 1: Emotion Arc Prediction with GRU
Predicts emotion trajectory and detects inflection points
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from models.multitask_emotion_model import PluTchikMultiTaskModel, EMOTION_CLASSES

class EmotionArcPredictor(nn.Module):
    """
    GRU-based sequence model that predicts emotion distribution for next turn
    based on previous K turns' CLS embeddings.
    
    Architecture:
    - Takes sequence of CLS embeddings (128-dim) from previous turns
    - 2-layer GRU with hidden=256
    - Outputs predicted emotion distribution for turn K+1
    - Can detect inflection points via KL divergence
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, 
                 num_emotions=32, dropout=0.3, max_seq_len=10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention layer to weight importance of each turn
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Emotion prediction head
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
        
        # Inflection point detector (binary: will there be a major shift?)
        self.inflection_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, cls_embeddings, lengths=None):
        """
        Args:
            cls_embeddings: Tensor of shape (batch, seq_len, 128) - CLS embeddings from base model
            lengths: Optional tensor of actual sequence lengths (for padding mask)
        
        Returns:
            dict with:
                - emotion_logits: (batch, 32) predicted emotion distribution
                - inflection_logit: (batch, 1) probability of inflection point
                - attention_weights: (batch, seq_len) which turns mattered most
                - gru_output: (batch, hidden_dim) final GRU state
        """
        batch_size = cls_embeddings.size(0)
        seq_len = cls_embeddings.size(1)
        
        # Run through GRU
        gru_out, hidden = self.gru(cls_embeddings)  # gru_out: (batch, seq_len, hidden)
        
        # Attention over sequence
        attn_scores = self.attention(gru_out).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding if lengths provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=cls_embeddings.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum of GRU outputs
        context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)  # (batch, hidden)
        
        # Predictions
        emotion_logits = self.emotion_head(context)
        inflection_logit = self.inflection_head(context)
        
        return {
            'emotion_logits': emotion_logits,
            'inflection_logit': inflection_logit,
            'attention_weights': attn_weights,
            'gru_output': context
        }
    
    def predict_next_emotion(self, cls_sequence):
        """
        Convenience method for inference: predict next emotion from sequence of CLS embeddings
        
        Args:
            cls_sequence: List or array of CLS embeddings from previous turns
        
        Returns:
            dict with predicted emotion, confidence, and inflection probability
        """
        self.eval()
        
        # Convert to tensor and add batch dimension
        if isinstance(cls_sequence, list):
            cls_tensor = torch.stack(cls_sequence).unsqueeze(0)
        elif isinstance(cls_sequence, np.ndarray):
            cls_tensor = torch.from_numpy(cls_sequence).unsqueeze(0)
        else:
            cls_tensor = cls_sequence.unsqueeze(0) if cls_sequence.dim() == 2 else cls_sequence
        
        with torch.no_grad():
            outputs = self.forward(cls_tensor)
            
            # Emotion prediction
            emotion_probs = torch.softmax(outputs['emotion_logits'], dim=-1)[0]
            top_idx = emotion_probs.argmax().item()
            confidence = emotion_probs[top_idx].item()
            
            # Inflection probability
            inflection_prob = torch.sigmoid(outputs['inflection_logit']).item()
            
            # Get attention weights to see which turns mattered
            attn_weights = outputs['attention_weights'][0].cpu().numpy()
        
        return {
            'predicted_emotion': EMOTION_CLASSES[top_idx],
            'confidence': round(confidence, 4),
            'all_emotions': {EMOTION_CLASSES[i]: round(p.item(), 4) for i, p in enumerate(emotion_probs)},
            'inflection_probability': round(inflection_prob, 4),
            'turn_attention': [round(w, 4) for w in attn_weights],
            'is_inflection_likely': inflection_prob > 0.5
        }


def train_arc_predictor(base_model_path, train_data_path, epochs=10, lr=1e-3):
    """
    Train the arc predictor using existing dialogue data
    
    The training signal comes from the existing dataset:
    - For each conversation, use turns 0..T-1 to predict turn T's emotion
    - No additional annotation needed!
    """
    import json
    from transformers import BertTokenizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base model to extract CLS embeddings
    base_model = PluTchikMultiTaskModel().to(device)
    checkpoint = torch.load(base_model_path, map_location=device, weights_only=True)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize arc predictor
    arc_model = EmotionArcPredictor().to(device)
    optimizer = torch.optim.Adam(arc_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Load training data
    with open(train_data_path) as f:
        train_data = [json.loads(line) for line in f]
    
    # Group by conversation
    conversations = {}
    for item in train_data:
        conv_id = item['conversation_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(item)
    
    # Sort each conversation by turn number
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['turn_number'])
    
    print(f"Loaded {len(conversations)} conversations for arc training")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_samples = 0
        
        for conv_id, turns in conversations.items():
            if len(turns) < 2:  # Need at least 2 turns for arc prediction
                continue
            
            # Extract CLS embeddings for each turn
            cls_embeddings = []
            labels = []
            
            for turn_data in turns:
                # Encode turn text
                inputs = tokenizer.encode_plus(
                    turn_data['text'],
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                # Get CLS embedding from base model
                with torch.no_grad():
                    outputs = base_model(inputs['input_ids'], inputs['attention_mask'])
                    cls_emb = outputs['pooled_output'].squeeze(0).cpu()
                
                cls_embeddings.append(cls_emb)
                labels.append(EMOTION_CLASSES.index(turn_data['emotion']))
            
            # Create sequences: use turns 0..T-1 to predict turn T
            for target_idx in range(1, len(cls_embeddings)):
                seq_embeddings = torch.stack(cls_embeddings[:target_idx]).unsqueeze(0).to(device)
                target_label = torch.tensor([labels[target_idx]], dtype=torch.long).to(device)
                
                # Forward pass
                outputs = arc_model(seq_embeddings)
                
                # Compute loss
                loss = criterion(outputs['emotion_logits'], target_label)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
        
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return arc_model


if __name__ == "__main__":
    # Demo usage
    print("Emotion Arc Predictor - Phase 8 Extension 1")
    print("=" * 50)
    
    # Create dummy CLS sequence to test
    arc_model = EmotionArcPredictor()
    
    # Simulate 3 previous turns with random CLS embeddings (128-dim)
    dummy_cls_seq = [torch.randn(128) for _ in range(3)]
    
    result = arc_model.predict_next_emotion(dummy_cls_seq)
    
    print("\nPrediction from random CLS embeddings:")
    print(f"  Predicted emotion: {result['predicted_emotion']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Inflection likely: {result['is_inflection_likely']} ({result['inflection_probability']:.1%})")
    print(f"  Turn attention weights: {result['turn_attention']}")
    print("\nTop 5 emotions:")
    sorted_emotions = sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True)[:5]
    for emo, conf in sorted_emotions:
        print(f"  {emo}: {conf:.1%}")
