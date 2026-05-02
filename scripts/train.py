import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.multitask_emotion_model import PluTchikMultiTaskModel, EMOTION_CLASSES, INTENSITY_LABELS

EMOTION2IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}
INTENSITY2IDX = {i: idx for idx, i in enumerate(INTENSITY_LABELS)}

class PlutchikDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path) as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer.encode_plus(
            s['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        emotion_idx = EMOTION2IDX.get(s['emotion'], 0)
        intensity_idx = INTENSITY2IDX.get(s['intensity'], 1)
        sarcasm_label = 1.0 if s['sarcasm'] else 0.0
        iaa_score = s.get('iaa_score', 0.8)
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'emotion_label': torch.tensor(emotion_idx, dtype=torch.long),
            'sarcasm_label': torch.tensor(sarcasm_label, dtype=torch.float),
            'intensity_label': torch.tensor(intensity_idx, dtype=torch.long),
            'iaa_weight': torch.tensor(iaa_score, dtype=torch.float)
        }

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct_emotion = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['emotion_label'].to(device)
        sarcasm_labels = batch['sarcasm_label'].to(device)
        intensity_labels = batch['intensity_label'].to(device)
        iaa_weights = batch['iaa_weight'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels, sarcasm_labels, intensity_labels)
        
        # IAA-weighted loss
        loss = outputs['loss']
        weighted_loss = loss * iaa_weights.mean()
        
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item()
        
        # Accuracy tracking
        preds = outputs['emotion_logits'].argmax(dim=-1)
        correct_emotion += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_emotion / total
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Emotion Acc={accuracy:.4f}")
    return avg_loss, accuracy

def main():
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    print("Loading dataset...")
    train_dataset = PlutchikDataset('/workspace/plutchik_erc/data/train.jsonl', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print("Initializing model (tiny BERT: 2 layers, 128 hidden)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PluTchikMultiTaskModel().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters (~{total_params*4/1e6:.1f}MB)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    print(f"Training on {device} for 3 epochs...")
    best_acc = 0
    
    for epoch in range(1, 4):
        loss, acc = train_epoch(model, train_loader, optimizer, device, epoch)
        
        if acc > best_acc:
            best_acc = acc
            # Save checkpoint
            save_dir = Path('/workspace/plutchik_erc/my_plutchik_model')
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': acc
            }, save_dir / 'best_model.pt')
            print(f"✓ Saved best model (acc={acc:.4f})")
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: /workspace/plutchik_erc/my_plutchik_model/best_model.pt")

if __name__ == '__main__':
    main()
