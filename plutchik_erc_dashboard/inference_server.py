"""FastAPI Inference Server for Plutchik ERC."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Import model and codes
exec(open("train_minimal.py").read().split('if __name__')[0])

app = FastAPI(title="Plutchik ERC API")

# Load model
device = torch.device('cpu')
model = TinyModel()
ckpt = torch.load("plutchik_erc_dashboard/my_plutchik_model/best_model.pt", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Model loaded (val_acc={ckpt['val_acc']:.3f})")


class PredictRequest(BaseModel):
    text: str
    context: list[str] = []


def tokenize(text, max_len=32):
    ids = [ord(c) % 1000 for c in text[:max_len]]
    ids += [0] * (max_len - len(ids))
    mask = [1] * min(len(text), max_len) + [0] * max(0, max_len - len(text))
    return torch.tensor([ids]), torch.tensor([mask])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(req: PredictRequest):
    # Combine context + text
    full_text = " ".join(req.context[-2:] + [req.text]) if req.context else req.text
    
    inp, mask = tokenize(full_text)
    
    with torch.no_grad():
        out = model(inp.to(device), mask.to(device))
        
        emo_probs = torch.softmax(out['emotion_logits'], dim=1)[0]
        sar_probs = torch.softmax(out['sarcasm_logits'], dim=1)[0]
        intensity = out['intensity'][0].item()
        
        top_emotion_idx = emo_probs.argmax().item()
        top_emotion = EMOTION_CODES[top_emotion_idx]
        confidence = emo_probs[top_emotion_idx].item()
        
        # Build full distribution
        emotion_dist = {EMOTION_CODES[i]: round(emo_probs[i].item(), 4) for i in range(32)}
    
    return {
        "text": req.text,
        "primary_emotion": top_emotion,
        "confidence": round(confidence, 3),
        "intensity": round(intensity, 3),
        "sarcasm_probability": round(sar_probs[1].item(), 3),
        "emotion_distribution": emotion_dist,
        "ring_level": "intense" if intensity > 0.67 else ("mild" if intensity < 0.33 else "primary")
    }


@app.post("/predict/arc")
def predict_arc(conversation: list[str]):
    """Predict emotion trajectory across conversation turns."""
    trajectory = []
    
    for i, turn in enumerate(conversation):
        ctx = conversation[:i]
        result = predict(PredictRequest(text=turn, context=ctx))
        trajectory.append({
            "turn": i + 1,
            "text": turn[:50] + "..." if len(turn) > 50 else turn,
            "primary_emotion": result["primary_emotion"],
            "confidence": result["confidence"],
            "intensity": result["intensity"]
        })
    
    # Detect inflection points (large KL divergence)
    inflections = []
    for i in range(1, len(trajectory)):
        prev_emo = trajectory[i-1]["primary_emotion"]
        curr_emo = trajectory[i]["primary_emotion"]
        prev_conf = trajectory[i-1]["confidence"]
        curr_conf = trajectory[i]["confidence"]
        
        if prev_emo != curr_emo and abs(curr_conf - prev_conf) > 0.2:
            inflections.append({"turn": i+1, "from": prev_emo, "to": curr_emo})
    
    return {"trajectory": trajectory, "inflection_points": inflections}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
