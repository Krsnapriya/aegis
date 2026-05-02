"""Minimal Plutchik ERC - Fast training, CPU-friendly, demo-ready."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
import random
from pathlib import Path

# 32 emotion codes
EMOTION_CODES = [
    "sere", "acce", "dist", "pens", "bore", "inte", "anno", "opti",
    "joy", "trus", "fear", "surp", "sadn", "disg", "ange", "appr",
    "ecst", "admi", "terr", "amaz", "grie", "loat", "rage", "vigi",
    "love", "subm", "awe", "disa", "remo", "cont", "aggr", "supr"
]
CODE_TO_IDX = {c: i for i, c in enumerate(EMOTION_CODES)}


class TinyModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128, num_emotions=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.emotion_head = nn.Linear(64, num_emotions)
        self.sarcasm_head = nn.Linear(64, 2)
        self.intensity_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    
    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        out, _ = self.lstm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)
        h = self.fc(pooled)
        return {
            'emotion_logits': self.emotion_head(h),
            'sarcasm_logits': self.sarcasm_head(h),
            'intensity': self.intensity_head(h).squeeze(-1)
        }


class SimpleDataset(Dataset):
    def __init__(self, data_path, max_len=32):
        self.samples = []
        self.max_len = max_len
        with open(data_path, 'r') as f:
            for row in csv.DictReader(f):
                self.samples.append(row)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        # Simple char-level tokenization
        ids = [ord(c) % 1000 for c in s['text'][:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        mask = [1] * min(len(s['text']), self.max_len) + [0] * max(0, self.max_len - len(s['text']))
        
        emo_idx = CODE_TO_IDX.get(s['emotion_code'], 0)
        sarc = int(s['sarcasm'])
        intensity = (int(s['ring_level']) - 1) / 2.0
        iaa = float(s['iaa_score'])
        
        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'emotion_label': torch.tensor(emo_idx),
            'sarcasm_label': torch.tensor(sarc),
            'intensity_label': torch.tensor(intensity),
            'iaa_weight': torch.tensor(iaa)
        }


def train():
    print("=== Plutchik ERC Training (Minimal) ===")
    
    # Generate data if missing
    data_path = Path("plutchik_erc_dashboard/data/plutchik_dataset.csv")
    if not data_path.exists():
        print("Generating dataset...")
        from scripts.generate_data import generate_dataset
        generate_dataset(1500, str(data_path))
    
    device = torch.device('cpu')
    model = TinyModel().to(device)
    
    dataset = SimpleDataset(str(data_path))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [int(len(dataset)*0.85), len(dataset)-int(len(dataset)*0.85)])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    crit_ce = nn.CrossEntropyLoss(reduction='none')
    crit_mse = nn.MSELoss(reduction='none')
    
    best_acc = 0
    for epoch in range(3):
        model.train()
        for b in train_loader:
            inp = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            emo_lbl = b['emotion_label'].to(device)
            sar_lbl = b['sarcasm_label'].to(device)
            int_lbl = b['intensity_label'].to(device)
            iaa = b['iaa_weight'].to(device)
            
            out = model(inp, mask)
            loss = ((crit_ce(out['emotion_logits'], emo_lbl) + crit_ce(out['sarcasm_logits'], sar_lbl) + 0.5*crit_mse(out['intensity'], int_lbl)) * iaa).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for b in val_loader:
                inp = b['input_ids'].to(device)
                mask = b['attention_mask'].to(device)
                emo_lbl = b['emotion_label'].to(device)
                out = model(inp, mask)
                pred = out['emotion_logits'].argmax(dim=1)
                correct += (pred == emo_lbl).sum().item()
                total += len(emo_lbl)
        
        acc = correct / total
        print(f"Epoch {epoch+1}/3 - Val Acc: {acc:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            Path("plutchik_erc_dashboard/my_plutchik_model").mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': acc}, "plutchik_erc_dashboard/my_plutchik_model/best_model.pt")
            print(f"  ✓ Saved best model")
    
    print(f"\n✅ Training complete! Best val acc: {best_acc:.3f}")
    return model


if __name__ == "__main__":
    train()
