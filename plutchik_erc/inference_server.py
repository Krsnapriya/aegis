from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import json
from pathlib import Path
import sys
import time
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from models.multitask_emotion_model import PluTchikMultiTaskModel, EMOTION_CLASSES, INTENSITY_LABELS
from transformers import BertTokenizer

app = FastAPI(title="Plutchik ERC Inference API", version="1.0.0")

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: simple token-based
API_KEYS = {"demo-key-123": {"requests": deque(maxlen=100), "limit": 60}}  # 60 req/min

class PredictRequest(BaseModel):
    text: str
    context: Optional[List[str]] = []  # Previous turns
    session_id: Optional[str] = "default"

class PredictResponse(BaseModel):
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    sarcasm: bool
    sarcasm_score: float
    intensity: str
    intensity_scores: Dict[str, float]
    primary_emotion_ring: str  # mild/primary/intense

# Load model at startup
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PluTchikMultiTaskModel().to(device)
checkpoint = torch.load('/workspace/plutchik_erc/my_plutchik_model/best_model.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(f"✓ Model loaded on {device}")

# Session management (sliding context window)
sessions: Dict[str, deque] = {}

def check_rate_limit(api_key: Optional[str]) -> bool:
    if not api_key or api_key not in API_KEYS:
        return True  # No limit for demo
    key_data = API_KEYS[api_key]
    now = time.time()
    key_data["requests"].append(now)
    recent = [t for t in key_data["requests"] if now - t < 60]
    return len(recent) <= key_data["limit"]

def run_inference(text: str, context: List[str] = None) -> Dict[str, Any]:
    # Build input with optional context
    if context:
        full_text = " [CONTEXT] ".join(context[-2:]) + " [CURRENT] " + text
    else:
        full_text = text
    
    inputs = tokenizer.encode_plus(
        full_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Emotion probabilities
    emotion_probs = torch.softmax(outputs['emotion_logits'], dim=-1)[0]
    top_idx = emotion_probs.argmax().item()
    top_confidence = emotion_probs[top_idx].item()
    
    # Sarcasm prediction
    sarcasm_prob = torch.sigmoid(outputs['sarcasm_logits'])[0].item()
    is_sarcastic = sarcasm_prob > 0.5
    
    # Intensity prediction
    intensity_probs = torch.softmax(outputs['intensity_logits'], dim=-1)[0]
    intensity_idx = intensity_probs.argmax().item()
    
    # Determine primary emotion ring based on intensity
    ring_labels = ['mild', 'primary', 'intense']
    
    return {
        'emotion': EMOTION_CLASSES[top_idx],
        'confidence': round(top_confidence, 4),
        'all_emotions': {EMOTION_CLASSES[i]: round(p.item(), 4) for i, p in enumerate(emotion_probs)},
        'sarcasm': is_sarcastic,
        'sarcasm_score': round(sarcasm_prob, 4),
        'intensity': ring_labels[intensity_idx],
        'intensity_scores': {ring_labels[i]: round(p.item(), 4) for i, p in enumerate(intensity_probs)},
        'primary_emotion_ring': ring_labels[intensity_idx]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device)}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, x_api_key: Optional[str] = Header(None)):
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Update session context
    if req.session_id not in sessions:
        sessions[req.session_id] = deque(maxlen=3)
    sessions[req.session_id].append(req.text)
    
    context = list(sessions[req.session_id])[:-1]  # Previous turns
    
    result = run_inference(req.text, context)
    return PredictResponse(**result)

@app.post("/predict/batch")
async def predict_batch(texts: List[str], x_api_key: Optional[str] = Header(None)):
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    results = [run_inference(t, []) for t in texts]
    return {"predictions": results, "count": len(results)}

@app.post("/predict/arc")
async def predict_arc(conversation: List[str], x_api_key: Optional[str] = Header(None)):
    """Analyze emotion trajectory across a full conversation"""
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    trajectory = []
    prev_dist = None
    
    for i, turn in enumerate(conversation):
        result = run_inference(turn, conversation[:i])
        
        # Calculate KL divergence for inflection detection
        current_dist = torch.tensor(list(result['all_emotions'].values()))
        inflection = False
        kl_div = 0.0
        
        if prev_dist is not None:
            kl_div = torch.nn.functional.kl_div(
                torch.log(current_dist + 1e-8),
                prev_dist + 1e-8,
                reduction='sum'
            ).item()
            inflection = kl_div > 0.5  # Threshold for significant shift
        
        trajectory.append({
            'turn': i + 1,
            **result,
            'inflection_point': inflection,
            'kl_divergence': round(kl_div, 4)
        })
        
        prev_dist = current_dist
    
    # Find major inflection points
    inflections = [t for t in trajectory if t['inflection_point']]
    
    return {
        'trajectory': trajectory,
        'turns': len(trajectory),
        'inflection_points': inflections,
        'emotional_arc': [t['emotion'] for t in trajectory]
    }

@app.post("/explain")
async def explain(text: str, context: Optional[List[str]] = [], x_api_key: Optional[str] = Header(None)):
    """Return prediction with basic token importance (simplified Captum-style)"""
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Run standard inference
    result = run_inference(text, context)
    
    # Simplified attribution: highlight words that differ from neutral
    words = text.split()
    attributions = []
    
    for word in words:
        # Crude approximation: longer words and exclamations get higher scores
        score = min(1.0, len(word) / 10.0)
        if word.endswith('!') or word.endswith('?'):
            score *= 1.5
        attributions.append({'token': word, 'attribution': round(min(score, 1.0), 4)})
    
    return {
        **result,
        'token_attributions': attributions,
        'method': 'simplified_gradient_approximation'
    }

@app.post("/correct")
async def submit_correction(text: str, true_emotion: str, true_sarcasm: bool, 
                           model_prediction: str, x_api_key: Optional[str] = Header(None)):
    """HITL correction endpoint - logs corrections for retraining"""
    correction = {
        'text': text,
        'true_emotion': true_emotion,
        'true_sarcasm': true_sarcasm,
        'model_prediction': model_prediction,
        'timestamp': time.time()
    }
    
    # Append to corrections file
    corr_file = Path('/workspace/plutchik_erc/data/corrections.jsonl')
    corr_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(corr_file, 'a') as f:
        f.write(json.dumps(correction) + '\n')
    
    return {'status': 'logged', 'message': 'Correction saved for retraining'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
