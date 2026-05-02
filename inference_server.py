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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models.multitask_emotion_model import PluTchikMultiTaskModel, EMOTION_CLASSES, INTENSITY_LABELS
from transformers import BertTokenizer
from captum.attr import IntegratedGradients

# Import the advanced engine
from advanced_engine import AdvancedPlutchikEngine

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

class DynamicAnalysisRequest(BaseModel):
    text: str
    session_id: Optional[str] = "default"
    user_baseline: Optional[Dict[str, Any]] = None

class DynamicAnalysisResponse(BaseModel):
    risk_level: str
    sarcasm_probability: float
    trajectory_forecast: List[List[float]]
    inflection_point: int
    reframe_suggestions: List[str]
    baseline_deviation: Optional[Dict[str, Any]]
    signals: List[str]

# Load model at startup
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PluTchikMultiTaskModel().to(device)
checkpoint = torch.load('/workspace/my_plutchik_model/best_model.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(f"✓ Model loaded on {device}")

# Session management (sliding context window)
sessions: Dict[str, deque] = {}

# Initialize advanced engine
advanced_engine = AdvancedPlutchikEngine(device=device)

# Captum Integrated Gradients for explainability
ig = None

def init_captum():
    global ig
    if ig is None:
        ig = IntegratedGradients(model)

def get_token_attributions(text: str, context: List[str] = None, target_emotion_idx: int = None) -> List[Dict]:
    """Use Captum Integrated Gradients to get token-level attributions"""
    init_captum()
    
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
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # If no target emotion specified, use the predicted one
    if target_emotion_idx is None:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            emotion_probs = torch.softmax(outputs['emotion_logits'], dim=-1)[0]
            target_emotion_idx = emotion_probs.argmax().item()
    
    # Compute Integrated Gradients
    def predict_func(inputs_tensor):
        outputs = model(inputs_tensor, attention_mask)
        return outputs['emotion_logits'][:, target_emotion_idx]
    
    input_ids.requires_grad_(True)
    attributions, delta = ig.attribute(
        input_ids,
        baselines=input_ids.clone().fill_(tokenizer.pad_token_id),
        target=target_emotion_idx,
        return_convergence_delta=True
    )
    
    attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    # Map attributions to tokens
    token_attributions = []
    for i, (token, attr) in enumerate(zip(tokens, attributions)):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            token_attributions.append({
                'token': token.replace('##', ''),
                'attribution': round(float(abs(attr)), 4),
                'signed_attribution': round(float(attr), 4),
                'position': i
            })
    
    # Normalize attributions to 0-1 range
    if token_attributions:
        max_attr = max(t['attribution'] for t in token_attributions)
        if max_attr > 0:
            for t in token_attributions:
                t['attribution'] = round(t['attribution'] / max_attr, 4)
    
    return token_attributions

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

@app.post("/analyze/dynamic", response_model=DynamicAnalysisResponse)
async def analyze_dynamic(req: DynamicAnalysisRequest, x_api_key: Optional[str] = Header(None)):
    """
    Advanced dynamic analysis with trajectory forecasting, sarcasm detection, and reframing.
    Uses Neural ODEs for forecasting and multimodal incongruity for sarcasm.
    """
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Get current emotion from standard inference
    context = list(sessions.get(req.session_id, deque()))[:-1]
    base_result = run_inference(req.text, context)
    
    # Convert emotion dict to vector
    emotion_vector = [base_result['all_emotions'].get(e, 0.0) for e in EMOTION_CLASSES]
    
    # Build dialogue history for forecasting
    dialogue_history = []
    for turn_text in context:
        turn_result = run_inference(turn_text, [])
        turn_vector = [turn_result['all_emotions'].get(e, 0.0) for e in EMOTION_CLASSES]
        dialogue_history.append(turn_vector)
    
    # Run advanced analysis
    analysis = advanced_engine.analyze_dynamic(
        text=req.text,
        current_emotion_vector=emotion_vector,
        dialogue_history=dialogue_history,
        user_baseline=req.user_baseline
    )
    
    # Update session
    if req.session_id not in sessions:
        sessions[req.session_id] = deque(maxlen=3)
    sessions[req.session_id].append(req.text)
    
    return DynamicAnalysisResponse(
        risk_level=analysis['risk_level'],
        sarcasm_probability=analysis['incongruity']['sarcasm_probability'],
        trajectory_forecast=analysis['forecast']['trajectory'],
        inflection_point=analysis['forecast']['inflection_point_step'],
        reframe_suggestions=analysis['reframe_suggestions'],
        baseline_deviation=analysis['baseline_deviation'],
        signals=analysis['incongruity']['signals']
    )

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

@app.post("/compare")
async def compare_conversations(conv_a: List[str], conv_b: List[str], x_api_key: Optional[str] = Header(None)):
    """Compare two conversations side-by-side to see where emotional trajectories diverge"""
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Analyze both conversations
    arc_a = []
    arc_b = []
    
    for i, turn in enumerate(conv_a):
        result = run_inference(turn, conv_a[:i])
        arc_a.append({'turn': i + 1, 'text': turn, **result})
    
    for i, turn in enumerate(conv_b):
        result = run_inference(turn, conv_b[:i])
        arc_b.append({'turn': i + 1, 'text': turn, **result})
    
    # Calculate divergence at each turn
    comparison = []
    max_turns = max(len(arc_a), len(arc_b))
    
    for i in range(max_turns):
        turn_data = {'turn': i + 1}
        
        if i < len(arc_a):
            turn_data['conv_a'] = {
                'emotion': arc_a[i]['emotion'],
                'confidence': arc_a[i]['confidence'],
                'sarcasm': arc_a[i]['sarcasm'],
                'intensity': arc_a[i]['intensity']
            }
            dist_a = torch.tensor(list(arc_a[i]['all_emotions'].values()))
        else:
            turn_data['conv_a'] = None
            dist_a = None
        
        if i < len(arc_b):
            turn_data['conv_b'] = {
                'emotion': arc_b[i]['emotion'],
                'confidence': arc_b[i]['confidence'],
                'sarcasm': arc_b[i]['sarcasm'],
                'intensity': arc_b[i]['intensity']
            }
            dist_b = torch.tensor(list(arc_b[i]['all_emotions'].values()))
        else:
            turn_data['conv_b'] = None
            dist_b = None
        
        # Calculate Jensen-Shannon divergence between distributions
        if dist_a is not None and dist_b is not None:
            m = 0.5 * (dist_a + dist_b)
            js_div = 0.5 * (
                torch.nn.functional.kl_div(torch.log(dist_a + 1e-8), m + 1e-8, reduction='sum') +
                torch.nn.functional.kl_div(torch.log(dist_b + 1e-8), m + 1e-8, reduction='sum')
            ).item()
            turn_data['divergence'] = round(js_div, 4)
            turn_data['emotion_match'] = arc_a[i]['emotion'] == arc_b[i]['emotion']
        else:
            turn_data['divergence'] = None
            turn_data['emotion_match'] = None
    
    # Find key divergence points (where emotions differ significantly)
    high_divergence_turns = [t for t in comparison if t.get('divergence', 0) > 0.3 or t.get('emotion_match') == False]
    
    # Summary statistics
    summary = {
        'conv_a_length': len(arc_a),
        'conv_b_length': len(arc_b),
        'conv_a_avg_confidence': sum(t['confidence'] for t in arc_a) / len(arc_a) if arc_a else 0,
        'conv_b_avg_confidence': sum(t['confidence'] for t in arc_b) / len(arc_b) if arc_b else 0,
        'conv_a_sarcasm_rate': sum(1 for t in arc_a if t['sarcasm']) / len(arc_a) if arc_a else 0,
        'conv_b_sarcasm_rate': sum(1 for t in arc_b if t['sarcasm']) / len(arc_b) if arc_b else 0,
        'emotion_agreement_rate': sum(1 for t in comparison if t.get('emotion_match')) / len(comparison) if comparison else 0,
        'avg_divergence': sum(t.get('divergence', 0) for t in comparison if t.get('divergence') is not None) / len([t for t in comparison if t.get('divergence') is not None]) if comparison else 0
    }
    
    return {
        'comparison': comparison,
        'summary': summary,
        'key_divergence_points': high_divergence_turns,
        'conv_a_trajectory': arc_a,
        'conv_b_trajectory': arc_b
    }

@app.post("/explain")
async def explain(text: str, context: Optional[List[str]] = [], x_api_key: Optional[str] = Header(None)):
    """Return prediction with Captum Integrated Gradients token attributions"""
    if not check_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Run standard inference first to get prediction
    result = run_inference(text, context)
    
    # Get Captum token attributions
    try:
        token_attributions = get_token_attributions(text, context)
        
        # Also analyze context influence - which words in previous turns affected this prediction
        context_attributions = []
        if context:
            # Run IG on the full context + current to see which context tokens mattered
            full_context_text = " [CONTEXT] ".join(context[-2:])
            context_tokens = get_token_attributions(full_context_text, [], 
                                                    target_emotion_idx=EMOTION_CLASSES.index(result['emotion']))
            context_attributions = context_tokens[:20]  # Top 20 context tokens
        
        method = 'captum_integrated_gradients'
    except Exception as e:
        # Fallback to simple method if Captum fails
        words = text.split()
        token_attributions = [
            {'token': w, 'attribution': round(min(1.0, len(w) / 10.0), 4)} 
            for w in words
        ]
        context_attributions = []
        method = f'simplified_fallback_error:{str(e)[:50]}'
    
    return {
        **result,
        'token_attributions': token_attributions,
        'context_attributions': context_attributions,
        'method': method,
        'explanation_summary': {
            'top_positive_tokens': sorted(token_attributions, key=lambda x: x.get('signed_attribution', x['attribution']), reverse=True)[:5],
            'top_negative_tokens': sorted(token_attributions, key=lambda x: x.get('signed_attribution', -x['attribution']))[:5]
        }
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
