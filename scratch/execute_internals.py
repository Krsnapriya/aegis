import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.multitask_emotion_model import PluTchikMultiTaskModel
from utils.explainability_v2 import CaptumExplainer
from transformers import RobertaTokenizer
from utils.constants import PLUTCHIK, EMOTION_NAMES, NUM_EMOTIONS

def execute_internals():
    print("\n" + "="*60)
    print("🚀  PLUTCHIK ERC MODEL INTERNALS EXECUTION")
    print("="*60)
    
    # 1. Setup Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[SYSTEM] Initializing models on {device.upper()}...")
    
    # Initialize model (will use roberta-base and random head weights as best_model.pt is missing)
    model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)
    
    # Check if weights exist, otherwise warn user
    checkpoint_path = project_root / "my_plutchik_model" / "best_model.pt"
    if checkpoint_path.exists():
        print(f"[INFO] Loading custom weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print("[WARNING] best_model.pt not found. Using pre-trained RoBERTa backbone with random heads.")
        print("[WARNING] Emotion predictions will be stochastic, but internal logic is fully operational.")

    model.to(device)
    model.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    explainer = CaptumExplainer(model, tokenizer)
    
    # 2. Define Sample Texts
    text1 = "I am so happy and excited about this new project!"
    text2 = "I feel very sad and disappointed with the results."
    
    print(f"\n[DATA] Sample 1: \"{text1}\"")
    print(f"[DATA] Sample 2: \"{text2}\"")
    
    # 3. Extract Embeddings
    print("\n" + "-"*60)
    print("📍 STEP 1: CLS TOKEN EMBEDDINGS (High-Dimensional Latent Space)")
    print("-"*60)
    
    def get_embedding(text):
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
        return outputs['cls_embedding']
    
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    print(f"✅ Embedding 1 Vector Shape: {list(emb1.shape)} (768-dim RoBERTa Latent)")
    print(f"✅ Embedding 2 Vector Shape: {list(emb2.shape)}")
    print(f"📌 Embedding 1 (first 8 features): {emb1[0, :8].tolist()}")
    print(f"📌 Embedding 2 (first 8 features): {emb2[0, :8].tolist()}")
    
    # 4. Calculate Cosine Similarity
    print("\n" + "-"*60)
    print("📐 STEP 2: COSINE SIMILARITY (Semantic Distance)")
    print("-"*60)
    cos_sim = F.cosine_similarity(emb1, emb2)
    print(f"✅ Cosine Similarity Score: {cos_sim.item():.4f}")
    
    dist = 1.0 - cos_sim.item()
    relation = "SEMANTICALLY DISTANT" if dist > 0.5 else "SEMANTICALLY CLOSE"
    print(f"💡 Interpretation: {relation} (Angular Distance: {dist:.4f})")
    
    # 5. Token Attribution
    print("\n" + "-"*60)
    print("🔍 STEP 3: TOKEN ATTRIBUTIONS (Feature Importance via Integrated Gradients)")
    print("-"*60)
    
    # Get top predicted class for text1
    encoding1 = tokenizer(text1, return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs1 = model(encoding1['input_ids'], encoding1['attention_mask'])
    probs1 = torch.softmax(outputs1['emotion_logits'], dim=-1)
    top_class = torch.argmax(probs1).item()
    predicted_emotion = EMOTION_NAMES[top_class]
    confidence = probs1[0, top_class].item()
    
    print(f"🎯 Target Emotion: {predicted_emotion.upper()} (Confidence: {confidence:.2%})")
    print("⏳ Running Integrated Gradients (5 steps of Riemann approximation for SPEED)...")
    
    attributions = explainer.attribute_tokens(text1, target_class=top_class, n_steps=5)
    
    print("\n[TOP CONTRIBUTING TOKENS]")
    print(f"{'TOKEN':<15} | {'IMPORTANCE SCORE':<20}")
    print("-" * 38)
    
    # Filter out special tokens like <s> </s>
    filtered_attributions = [a for a in attributions if a['token'] not in ['<s>', '</s>', '<pad>']]
    sorted_attributions = sorted(filtered_attributions, key=lambda x: abs(x['score']), reverse=True)
    
    for attr in sorted_attributions[:12]:
        bar_len = int(abs(attr['score']) * 20)
        bar = ("+" if attr['score'] > 0 else "-") * bar_len
        print(f"{attr['token']:<15} | {attr['score']:>8.4f}  {bar}")

    print("\n" + "="*60)
    print("✅ INTERNAL EXECUTION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    execute_internals()
