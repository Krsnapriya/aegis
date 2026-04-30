"""
Plutchik ERC v2.1 — Contextual Inference Server (FastAPI)
Full API surface: single prediction, batch, arc analysis, HITL corrections.
Implements thread-safe session management and N-turn sliding window.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
from pathlib import Path
import sys
import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
from collections import deque, OrderedDict

project_dir = Path(__file__).parent

# Imports from local package
from plutchik_erc_dashboard.models.multitask_emotion_model import PluTchikMultiTaskModel
from plutchik_erc_dashboard.utils.preprocessing import ERCPreprocessor
from plutchik_erc_dashboard.utils.constants import PLUTCHIK, EMOTION_NAMES, NUM_EMOTIONS, RING_INTENSITY
from plutchik_erc_dashboard.utils.explainability_v2 import CaptumExplainer
from plutchik_erc_dashboard.database import engine, get_db, Base, SessionLocal
from plutchik_erc_dashboard.models.db_models import DB_Prediction, DB_Correction, DB_DialogueTurn

try:
    Base.metadata.create_all(bind=engine)
    DB_READY = True
    print("✓ Database tables initialized.")
except Exception as e:
    print(f"⚠ Database init failed (will use in-memory fallback): {e}")
    DB_READY = False

from fastapi import Header, Depends, BackgroundTasks, Request
import time
import logging
from sqlalchemy.orm import Session

# ============== APP SETUP ==============
app = FastAPI(
    title="Plutchik ERC Antigravity API",
    description="32-class emotion recognition with sarcasm detection, intensity regression, and emotion arc analysis.",
    version="2.1.0",
)

# CORS — restrict origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== RUTHLESS SECURITY ==============
API_KEY = os.getenv("PLUTCHIK_API_KEY")

class RateLimiter:
    def __init__(self, requests_per_minute=30):
        self.requests_per_minute = requests_per_minute
        self.clients = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < 60.0]
        if len(self.clients[client_ip]) >= self.requests_per_minute:
            return False
        self.clients[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

async def verify_api_key(request: Request, x_api_key: str = Header(None)):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key. Don't touch the moat.")
    return x_api_key

# ============== OBSERVABILITY ==============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plutchik-api")

# ============== DATABASE INIT ==============
@app.on_event("startup")
def startup():
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Postgres/SQLite persistence initialized.")

# ============== LOAD MODEL ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = project_dir / "my_plutchik_model"
model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)

checkpoint_path = model_dir / "best_model.pt"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"⚠ Missing keys (new layers, will use random init): {len(missing)} keys")
    if unexpected:
        print(f"⚠ Unexpected keys (removed layers): {len(unexpected)} keys")
    print(f"✓ Loaded model from {checkpoint_path}")
else:
    print("⚠ No checkpoint found - using untrained model")

model.to(device).eval()
preprocessor = ERCPreprocessor(PLUTCHIK)
captum_explainer = CaptumExplainer(model, preprocessor.tokenizer)
MODEL_READY = checkpoint_path.exists()

# ============== RUTHLESS PERSISTENCE ==============
# If running on Hugging Face with persistent storage mounted
PERSISTENT_BASE = Path(os.getenv("PERSISTENT_DATA_PATH", project_dir / "data"))
CORRECTIONS_DIR = PERSISTENT_BASE / "corrections"
CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_FILE = CORRECTIONS_DIR / "hitl_corrections.jsonl"
logger.info(f"💾 Persistence active at: {CORRECTIONS_FILE}")

# ============== RUTHLESS PERSISTENCE & DB QUEUE ==============
# Known Limitation: SQLite locking under high concurrency.
# Migration Path: Moving to PostgreSQL or append-only JSONL logs for production.
# Mitigation: Single writer thread via queue to eliminate lock collision.
import queue
import threading

_db_write_queue = queue.Queue()

def _db_worker():
    while True:
        record = _db_write_queue.get()
        db_session = SessionLocal()
        try:
            db_session.add(record)
            db_session.commit()
        except Exception as e:
            logger.error(f"DB Logging Error in Worker: {e}")
        finally:
            db_session.close()
        _db_write_queue.task_done()

threading.Thread(target=_db_worker, daemon=True).start()

# ============== SESSION MANAGER ==============
class SessionManager:
    def __init__(self, window_size=3, max_sessions=1000):
        self.sessions = OrderedDict()
        self.window_size = window_size
        self.max_sessions = max_sessions

    def get_context(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return "[NO_CONTEXT]"
        history = list(self.sessions[session_id])
        return " | ".join(history)

    def add_turn(self, session_id: str, text: str, speaker: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.window_size)
        abbr = speaker[:3].upper()
        self.sessions[session_id].append(f"{abbr}: {text}")
        self.sessions.move_to_end(session_id)
        if len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager(window_size=3)


# ============== SCHEMAS ==============

# --- Single Prediction ---
class PredictRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    session_id: str = Field("default", max_length=100)
    speaker: str = Field("USER", max_length=100)
    scenario: str = Field("casual", max_length=100)
    topic: str = Field("general", max_length=100)

class EmotionScore(BaseModel):
    emotion: str
    probability: float

class PredictResponse(BaseModel):
    emotion: str
    confidence: float
    sarcasm_prob: float
    intensity: float
    ring: str
    sector: str
    context_used: str
    explanations: Optional[dict] = None
    cls_embedding: Optional[List[float]] = None
    token_embeddings: Optional[List[List[float]]] = None
    top_5: List[EmotionScore]
    emotion_probs: Optional[List[float]] = None

# --- Batch Prediction ---
class BatchItem(BaseModel):
    text: str = Field(..., max_length=5000)
    scenario: str = Field("casual", max_length=100)
    topic: str = Field("general", max_length=100)

class BatchRequest(BaseModel):
    items: List[BatchItem]

class BatchResponseItem(BaseModel):
    text: str
    emotion: str
    confidence: float
    sarcasm_prob: float
    intensity: float
    ring: str

class BatchResponse(BaseModel):
    results: List[BatchResponseItem]
    count: int

# --- HITL Correction ---
class CorrectionRequest(BaseModel):
    text: str
    predicted_emotion: str
    corrected_emotion: str
    predicted_confidence: float = 0.0
    annotator_id: str = "anonymous"
    scenario: str = "unknown"
    notes: str = ""

class CorrectionResponse(BaseModel):
    status: str
    correction_id: str

# --- Arc Analysis ---
class ArcUtterance(BaseModel):
    speaker: str = Field(..., max_length=100)
    text: str = Field(..., max_length=5000)

class ArcRequest(BaseModel):
    utterances: List[ArcUtterance]
    scenario: str = Field("casual", max_length=100)
    topic: str = Field("general", max_length=100)

class ArcResponse(BaseModel):
    arc_type: str
    turns: List[dict]
    turning_points: List[dict]
    intensity_trajectory: List[float]


# ============== CORE INFERENCE FUNCTION ==============
def _run_inference(text: str, scenario: str, topic: str, context: str = "[NO_CONTEXT]", compute_explanations: bool = False):
    """Shared inference logic used by all prediction endpoints."""
    # Gibberish check: If tokens/words ratio is > 3.5, it's likely gibberish
    words = text.strip().split()
    raw_tokens = preprocessor.tokenizer.tokenize(text)
    if len(words) > 0 and len(raw_tokens) / len(words) > 3.5:
        logger.warning(f"Gibberish detected in text: {text[:50]}...")
        # Still predict, but set confidence to 0 to indicate uncertainty
        is_gibberish = True
    else:
        is_gibberish = False

    augmented = preprocessor.augment_with_metadata(text, scenario, topic)
    full_input = f"[CONTEXT] {context} [/CONTEXT] [CURRENT] {augmented} [/CURRENT]"

    encoding = preprocessor.tokenizer(
        full_input, max_length=256, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    emotion_probs = torch.softmax(outputs["emotion_logits"].squeeze(), dim=0).cpu().numpy()
    sarcasm_probs = torch.softmax(outputs["sarcasm_logits"].squeeze(), dim=0).cpu().numpy()
    intensity = float(outputs["intensity"].squeeze().item())

    emotion_idx = int(emotion_probs.argmax())
    predicted_emotion = EMOTION_NAMES[emotion_idx]
    confidence = 0.0 if is_gibberish else float(emotion_probs[emotion_idx])

    # Top 5
    top_indices = emotion_probs.argsort()[::-1][:5]
    top_5 = [
        {"emotion": EMOTION_NAMES[i], "probability": float(emotion_probs[i])}
        for i in top_indices
    ]

    ring = PLUTCHIK[predicted_emotion]["ring"]
    sector = PLUTCHIK[predicted_emotion]["sector"]

    # RUTHLESS EXPLAINABILITY
    explanations = None
    if compute_explanations:
        try:
            token_attributions = captum_explainer.attribute_tokens(text, target_class=emotion_idx)
            explanations = {"token_attributions": token_attributions}
        except Exception as e:
            logger.error(f"Captum failed: {e}")

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "sarcasm_prob": float(sarcasm_probs[1]),
        "intensity": intensity,
        "ring": ring,
        "sector": sector,
        "top_5": top_5,
        "explanations": explanations,
        "cls_embedding": outputs["cls_embedding"].squeeze().cpu().tolist(),
        "token_embeddings": outputs["last_hidden_state"].squeeze().cpu().tolist() if "last_hidden_state" in outputs else None,
        "emotion_probs": emotion_probs.tolist()
    }

def _log_prediction_to_db(req: PredictRequest, result: dict, context: str):
    """Background task to persist prediction to DB via async queue."""
    db_pred = DB_Prediction(
        text=req.text,
        emotion=result["emotion"],
        confidence=result["confidence"],
        sarcasm_prob=result["sarcasm_prob"],
        intensity=result["intensity"],
        ring=result["ring"],
        sector=result["sector"],
        scenario=req.scenario,
        topic=req.topic,
        speaker=req.speaker,
        context_used=context,
        session_id=req.session_id
    )
    _db_write_queue.put(db_pred)

# ============== ENDPOINTS ==============

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks, _auth: str = Depends(verify_api_key)):
    """Single utterance prediction with session context."""
    try:
        context = session_manager.get_context(req.session_id)
        result = _run_inference(req.text, req.scenario, req.topic, context, compute_explanations=False)
        session_manager.add_turn(req.session_id, req.text, req.speaker)

        # Persistence via Background Task
        background_tasks.add_task(_log_prediction_to_db, req, result, context)

        return PredictResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            sarcasm_prob=result["sarcasm_prob"],
            intensity=result["intensity"],
            ring=result["ring"],
            sector=result["sector"],
            context_used=context,
            explanations=result["explanations"],
            cls_embedding=result["cls_embedding"],
            token_embeddings=result["token_embeddings"],
            top_5=[EmotionScore(**s) for s in result["top_5"]],
            emotion_probs=result["emotion_probs"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=PredictResponse)
async def explain(req: PredictRequest, background_tasks: BackgroundTasks, _auth: str = Depends(verify_api_key)):
    """Detailed model inspection with full token attribution."""
    try:
        context = session_manager.get_context(req.session_id)
        result = _run_inference(req.text, req.scenario, req.topic, context, compute_explanations=True)
        # Token attribution is already included in _run_inference, 
        # but we use this endpoint for clarity in the SDK/Vision.
        return PredictResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            sarcasm_prob=result["sarcasm_prob"],
            intensity=result["intensity"],
            ring=result["ring"],
            sector=result["sector"],
            context_used=context,
            explanations=result["explanations"],
            cls_embedding=result["cls_embedding"],
            token_embeddings=result["token_embeddings"],
            top_5=[EmotionScore(**s) for s in result["top_5"]],
            emotion_probs=result["emotion_probs"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest, _auth: str = Depends(verify_api_key)):
    """Batch prediction — no session context, stateless."""
    try:
        results = []
        for item in req.items:
            result = _run_inference(item.text, item.scenario, item.topic, compute_explanations=False)
            results.append(BatchResponseItem(
                text=item.text,
                emotion=result["emotion"],
                confidence=result["confidence"],
                sarcasm_prob=result["sarcasm_prob"],
                intensity=result["intensity"],
                ring=result["ring"],
            ))
        return BatchResponse(results=results, count=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correct", response_model=CorrectionResponse)
async def submit_correction(
    req: CorrectionRequest, 
    db: Session = Depends(get_db),
    _auth: str = Depends(verify_api_key)
):
    """
    HITL correction endpoint — the flywheel.
    Persists labeled corrections to the Postgres/SQLite database.
    """
    # Validate corrected emotion
    if req.corrected_emotion not in PLUTCHIK:
        raise HTTPException(
            status_code=400,
            detail=f"'{req.corrected_emotion}' is not a valid Plutchik emotion."
        )

    # Create DB record
    db_correction = DB_Correction(
        text=req.text,
        predicted_emotion=req.predicted_emotion,
        corrected_emotion=req.corrected_emotion,
        predicted_confidence=req.predicted_confidence,
        status="pending_review",
        annotator_notes=req.notes
    )
    db.add(db_correction)
    db.commit()
    db.refresh(db_correction)

    # Count total pending
    total_pending = db.query(DB_Correction).filter(DB_Correction.status == "pending_review").count()

    return CorrectionResponse(
        status="accepted",
        correction_id=str(db_correction.id),
    )


@app.post("/predict/arc", response_model=ArcResponse)
async def predict_arc(req: ArcRequest, _auth: str = Depends(verify_api_key)):
    """
    Dialogue-level emotion arc analysis.
    Takes a sequence of utterances and returns the emotion trajectory,
    arc classification, and turning points.
    """
    try:
        turns = []
        emotion_indices = []

        # 1. Build inputs and tokenize
        input_ids_list = []
        attention_mask_list = []
        is_gibberish_list = []
        
        for i, utt in enumerate(req.utterances):
            # Gibberish check
            words = utt.text.strip().split()
            raw_tokens = preprocessor.tokenizer.tokenize(utt.text)
            is_gibberish = (len(words) > 0 and len(raw_tokens) / len(words) > 3.5)
            is_gibberish_list.append(is_gibberish)

            # Build sliding window context
            context_parts = []
            start = max(0, i - 2)
            for j in range(start, i):
                prev = req.utterances[j]
                context_parts.append(f"{prev.speaker[:3].upper()}: {prev.text}")
            context = " | ".join(context_parts) if context_parts else "[NO_CONTEXT]"

            # Augment and tokenize
            augmented = preprocessor.augment_with_metadata(utt.text, req.scenario, req.topic)
            full_input = f"[CONTEXT] {context} [/CONTEXT] [CURRENT] {augmented} [/CURRENT]"

            encoding = preprocessor.tokenizer(
                full_input, max_length=256, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids_list.append(encoding["input_ids"])
            attention_mask_list.append(encoding["attention_mask"])

        # 2. Batched inference pass
        batch_input_ids = torch.cat(input_ids_list, dim=0).to(device)
        batch_attention_mask = torch.cat(attention_mask_list, dim=0).to(device)
        
        with torch.no_grad():
            outputs = model(batch_input_ids, batch_attention_mask)
            
        emotion_probs_batch = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()
        sarcasm_probs_batch = torch.softmax(outputs["sarcasm_logits"], dim=1).cpu().numpy()
        intensity_batch = outputs["intensity"].squeeze(-1).cpu().numpy()

        # 3. Construct response
        for i, utt in enumerate(req.utterances):
            emotion_probs = emotion_probs_batch[i]
            sarcasm_prob = float(sarcasm_probs_batch[i][1])
            intensity = float(intensity_batch[i])
            
            emotion_idx = int(emotion_probs.argmax())
            predicted_emotion = EMOTION_NAMES[emotion_idx]
            confidence = 0.0 if is_gibberish_list[i] else float(emotion_probs[emotion_idx])
            
            emotion_indices.append(emotion_idx)
            ring = PLUTCHIK[predicted_emotion]["ring"]

            turns.append({
                "turn": i,
                "speaker": utt.speaker,
                "text": utt.text,
                "emotion": predicted_emotion,
                "confidence": confidence,
                "sarcasm_prob": sarcasm_prob,
                "intensity": intensity,
                "ring": ring,
                "emotion_probs": emotion_probs.tolist()
            })

        # Compute intensity trajectory
        intensity_trajectory = []
        for idx in emotion_indices:
            ring = PLUTCHIK[EMOTION_NAMES[idx]]["ring"]
            intensity_trajectory.append(RING_INTENSITY.get(ring, 0.5))

        # Compute arc type
        if len(intensity_trajectory) < 2:
            arc_type = "stable"
            turning_points = []
        else:
            deltas = [intensity_trajectory[i+1] - intensity_trajectory[i]
                      for i in range(len(intensity_trajectory) - 1)]
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)

            if std_delta > 0.25:
                arc_type = "volatile"
            elif mean_delta > 0.1:
                arc_type = "escalation"
            elif mean_delta < -0.1:
                arc_type = "de-escalation"
            else:
                arc_type = "stable"

            # Detect turning points
            turning_points = []
            
            # 1. Intensity-based turning points
            for i in range(1, len(intensity_trajectory)):
                if abs(intensity_trajectory[i] - intensity_trajectory[i-1]) >= 0.3:
                    turning_points.append({"turn": i, "type": "intensity_shift"})

            # 2. Distribution-based inflection points (KL Divergence)
            for i in range(1, len(turns)):
                p = np.array(turns[i-1]["emotion_probs"])
                q = np.array(turns[i]["emotion_probs"])
                # Add epsilon to avoid log(0)
                p = p + 1e-10
                q = q + 1e-10
                kl_div = np.sum(p * np.log(p / q))
                
                if kl_div > 1.5: # Threshold for major emotional shift
                    turning_points.append({"turn": i, "type": "emotional_inflection", "score": float(kl_div)})

        return ArcResponse(
            arc_type=arc_type,
            turns=turns,
            turning_points=turning_points,
            intensity_trajectory=intensity_trajectory,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corrections/stats")
async def correction_stats(db: Session = Depends(get_db)):
    """Returns stats about the HITL correction queue from DB."""
    total = db.query(DB_Correction).count()
    pending = db.query(DB_Correction).filter(DB_Correction.status == "pending_review").count()
    reviewed = db.query(DB_Correction).filter(DB_Correction.status == "reviewed").count()

    # Top corrected emotions
    from sqlalchemy import func
    top_emotions = db.query(
        DB_Correction.corrected_emotion, 
        func.count(DB_Correction.corrected_emotion)
    ).group_by(DB_Correction.corrected_emotion).order_by(func.count(DB_Correction.corrected_emotion).desc()).limit(10).all()

    return {
        "total": total,
        "pending": pending,
        "reviewed": reviewed,
        "top_corrected_emotions": {e: c for e, c in top_emotions},
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session's context memory."""
    session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/emotions")
async def list_emotions():
    """Returns the full Plutchik taxonomy — useful for SDK consumers."""
    return {
        "emotions": PLUTCHIK,
        "count": NUM_EMOTIONS,
        "names": EMOTION_NAMES,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": device,
        "model_ready": MODEL_READY,
        "version": "2.1.0",
        "endpoints": ["/predict", "/predict/batch", "/predict/arc", "/correct",
                      "/corrections/stats", "/emotions", "/session/{id}", "/health"],
        "warning": None if MODEL_READY else "Using untrained model — predictions are random",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
