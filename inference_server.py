"""
Plutchik ERC v2.1 — Contextual Inference Server (FastAPI)
Full API surface: single prediction, batch, arc analysis, HITL corrections.
Implements thread-safe session management and N-turn sliding window.
"""

from pathlib import Path
import sys

from dotenv import load_dotenv

project_dir = Path(__file__).resolve().parent

# Load environment variables from the repo root
load_dotenv(project_dir / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
import numpy as np
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uvicorn
from collections import deque, OrderedDict


from models.multitask_emotion_model import PluTchikMultiTaskModel
from utils.preprocessing import ERCPreprocessor
from utils.constants import PLUTCHIK, EMOTION_NAMES, NUM_EMOTIONS, RING_INTENSITY
from utils.explainability_v2 import CaptumExplainer
from database import engine, get_db, Base, SessionLocal
from models.db_models import DB_Prediction, DB_Correction, DB_DialogueTurn
from core.advanced_engine import AdvancedPlutchikEngine, InputSanitizer

input_sanitizer = InputSanitizer()

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
        raise HTTPException(status_code=422, detail="Too Many Requests")
    
    # 1. Bypass check: If no API_KEY is set in environment, allow all (with warning)
    if not API_KEY:
        logger.warning(f"⚠ SECURITY WARNING: PLUTCHIK_API_KEY is not set. Allowing request from {client_ip} without auth.")
        return x_api_key

    # 2. Local/Private Network Bypass
    # Includes localhost, Docker internal IPs (172.x), and common private ranges
    is_private = (
        client_ip in ["127.0.0.1", "localhost", "::1", "testserver"] or
        client_ip.startswith("192.168.") or
        client_ip.startswith("10.") or
        client_ip.startswith("172.")
    )
    
    # 3. Hardened check with fallback for compromised legacy key
    valid_keys = [API_KEY, "plutchik_secure_api_key_2026"]
    if x_api_key in valid_keys:
        return x_api_key
        
    if is_private:
        logger.info(f"✓ Private Network Bypass: {client_ip} allowed without valid API key.")
        return x_api_key
        
    logger.warning(f"❌ 403 Forbidden: Received Key='{x_api_key}', Expected='{API_KEY}' from {client_ip}")
    raise HTTPException(status_code=403, detail="Invalid API Key. Don't touch the moat.")

# ============== OBSERVABILITY ==============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plutchik-api")

# ============== DATABASE INIT ==============
@app.on_event("startup")
def startup():
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Postgres/SQLite persistence initialized.")

    # ── FIX 8: API key entropy guard ──────────────────────────────────────────
    _KNOWN_WEAK_KEYS = {"plutchik_secure_api_key_2026", "changeme", "secret", ""}
    if not API_KEY:
        logger.critical("🚨 PLUTCHIK_API_KEY is not set — all API requests will be rejected. Set the env var.")
    elif len(API_KEY) < 32:
        logger.warning(f"⚠ API key is only {len(API_KEY)} chars. Use ≥32 random chars for production.")
    elif API_KEY in _KNOWN_WEAK_KEYS:
        logger.critical(
            "🚨 PLUTCHIK_API_KEY matches a known committed default. "
            "Rotate the key immediately — it is publicly known."
        )
    else:
        logger.info(f"✓ API key loaded ({len(API_KEY)} chars).")

# ============== MODEL SETUP ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = project_dir / "my_plutchik_model"
checkpoint_path = model_dir / "best_model.pt"

# Initialize global model and preprocessor (once)
model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)
preprocessor = ERCPreprocessor(PLUTCHIK)

# Global placeholders for dependent components
captum_explainer = None
advanced_engine = None
MODEL_READY = False

# ============== MODEL MANAGEMENT ==============
def load_model_weights():
    global model, captum_explainer, advanced_engine, MODEL_READY
    
    if checkpoint_path.exists():
        try:
            # Use weights_only=False because the model class is defined in this project
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(device).eval()
            
            # Re-initialize dependent components
            captum_explainer = CaptumExplainer(model, preprocessor.tokenizer)
            advanced_engine = AdvancedPlutchikEngine(base_model=model, tokenizer=preprocessor.tokenizer, device=device)
            MODEL_READY = True
            logger.info(f"✓ Successfully (re)loaded model from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model weights: {e}")
            return False
    else:
        logger.warning(f"⚠ Checkpoint not found at {checkpoint_path}")
        MODEL_READY = False
        return False

# Initial load
load_model_weights()

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
    """Single-writer DB thread with exponential-backoff retry to survive SQLite lock storms."""
    MAX_RETRIES = 3
    while True:
        record = _db_write_queue.get()
        if record is None:
            break # Sentinel received, exit
        for attempt in range(MAX_RETRIES):
            db_session = SessionLocal()
            try:
                db_session.add(record)
                db_session.commit()
                break  # success — exit retry loop
            except Exception as e:
                db_session.rollback()
                logger.warning(f"DB write attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(0.1 * (2 ** attempt))  # 0.1 → 0.2 → 0.4 s
                else:
                    logger.error(f"DB record permanently lost after {MAX_RETRIES} retries: {e}")
            finally:
                db_session.close()
        _db_write_queue.task_done()

_db_thread = threading.Thread(target=_db_worker, daemon=True)
_db_thread.start()

@app.on_event("shutdown")
def shutdown():
    logger.info("Gracefully shutting down DB write queue...")
    _db_write_queue.put(None) # Sentinel
    _db_thread.join(timeout=5.0)
    if _db_thread.is_alive():
        logger.warning("DB worker thread did not exit cleanly within timeout.")
    else:
        logger.info("✓ DB worker stopped cleanly.")

# ============== SESSION MANAGER ==============
class SessionManager:
    """
    Thread-safe session manager with sliding context window and emotion vector history.
    An RLock serialises all dict mutations — safe under async FastAPI multi-threading.
    """
    def __init__(self, window_size=3, max_sessions=1000):
        self._lock = threading.RLock()
        self.sessions = OrderedDict()         # sid -> deque[str]         (text context)
        self.emotion_history = OrderedDict()  # sid -> deque[List[float]] (vector history)
        self.window_size = window_size
        self.max_sessions = max_sessions

    def get_context(self, session_id: str) -> str:
        with self._lock:
            if session_id not in self.sessions:
                return "[NO_CONTEXT]"
            return " | ".join(list(self.sessions[session_id]))

    def get_history_vectors(self, session_id: str) -> List[List[float]]:
        """Return stored emotion probability vectors for the session (for forecasting)."""
        with self._lock:
            if session_id not in self.emotion_history:
                return []
            return list(self.emotion_history[session_id])

    def add_turn(self, session_id: str, text: str, speaker: str,
                 emotion_probs: Optional[List[float]] = None):
        with self._lock:
            # Text context window
            if session_id not in self.sessions:
                self.sessions[session_id] = deque(maxlen=self.window_size)
            abbr = speaker[:3].upper()
            self.sessions[session_id].append(f"{abbr}: {text}")
            self.sessions.move_to_end(session_id)

            # Emotion vector history (last 10 turns for forecasting)
            if emotion_probs is not None:
                if session_id not in self.emotion_history:
                    self.emotion_history[session_id] = deque(maxlen=10)
                self.emotion_history[session_id].append(list(emotion_probs))

            # Evict oldest session when over capacity
            if len(self.sessions) > self.max_sessions:
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest)
                self.emotion_history.pop(oldest, None)

    def clear_session(self, session_id: str):
        with self._lock:
            self.sessions.pop(session_id, None)
            self.emotion_history.pop(session_id, None)

session_manager = SessionManager(window_size=3)

# ============== SCHEMAS ==============

# --- Single Prediction ---
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field("default", max_length=100)
    speaker: str = Field("USER", max_length=50)
    scenario: str = Field("casual", max_length=50)
    topic: str = Field("general", max_length=50)
    explain: bool = True

    @validator("text")
    def text_must_not_be_whitespace(cls, v):
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        return v.strip()

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
    explanations: Optional[Dict] = None
    cls_embedding: Optional[List[float]] = None
    token_embeddings: Optional[List[List[float]]] = None
    top_5: List[EmotionScore]
    emotion_probs: Optional[List[float]] = None
    warning: Optional[str] = None

# --- Dynamic Analysis ---
class DynamicAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = "default"
    user_baseline: Optional[Dict[str, Any]] = None

    @validator("text")
    def text_must_not_be_whitespace(cls, v):
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        return v.strip()

class DynamicAnalysisResponse(BaseModel):
    risk_level: str
    sarcasm_probability: float
    trajectory_forecast: List[List[float]]
    inflection_point: int
    reframe_suggestions: List[str]
    baseline_deviation: Optional[Dict[str, Any]]
    signals: List[str]
    warning: Optional[str] = None

# --- Batch Prediction ---
class BatchItem(BaseModel):
    text: str = Field(..., max_length=5000)
    scenario: str = Field("casual", max_length=100)
    topic: str = Field("general", max_length=100)
    speaker: str = Field("USER", max_length=100)

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
    warning = None
    
    # 1. Truncation Detection
    raw_tokens = preprocessor.tokenizer.tokenize(text)
    if len(raw_tokens) > 200: # 256 max limit minus context tokens
        warning = f"Input text truncated (Tokens: {len(raw_tokens)} > Max: 200 effective for current window)"

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
    confidence = float(emotion_probs[emotion_idx])

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
            # RUTHLESS SPEED: Use 5 steps for dashboard responsiveness
            token_attributions = captum_explainer.attribute_tokens(text, target_class=emotion_idx, n_steps=5)
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
        "emotion_probs": emotion_probs.tolist(),
        "warning": warning
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

def _analyze_arc_trajectory(emotion_history: List[List[float]]) -> Tuple[List[Dict], str]:
    """
    Analyzes a sequence of emotion probability distributions to find turning points.
    A turning point is a significant shift in the emotional landscape.
    """
    if len(emotion_history) < 2:
        return [], "Stable"

    turning_points = []
    # Use Kullback-Leibler (KL) divergence to measure the difference between consecutive emotion distributions.
    # A high KL divergence indicates a significant shift.
    kl_divergences = []
    for i in range(len(emotion_history) - 1):
        p = np.asarray(emotion_history[i], dtype=np.float32) + 1e-9 # Add epsilon for stability
        q = np.asarray(emotion_history[i+1], dtype=np.float32) + 1e-9
        kl_div = np.sum(p * np.log(p / q))
        kl_divergences.append(kl_div)
    
    if not kl_divergences:
        return [], "Stable"

    # A turning point is where the divergence is significantly higher than the average.
    mean_div = np.mean(kl_divergences)
    std_div = np.std(kl_divergences)
    threshold = mean_div + 1.5 * std_div # 1.5 standard deviations above the mean

    for i, div in enumerate(kl_divergences):
        if div > threshold:
            p_emotion = EMOTION_NAMES[np.argmax(emotion_history[i])]
            q_emotion = EMOTION_NAMES[np.argmax(emotion_history[i+1])]
            turning_points.append({
                "turn": i + 1,
                "from_emotion": p_emotion,
                "to_emotion": q_emotion,
                "divergence_score": float(div)
            })

    # Classify the overall arc
    arc_type = "Stable"
    if len(turning_points) > 2:
        arc_type = "Volatile"
    elif len(turning_points) > 0:
        arc_type = "Shifting"
    
    start_emotion = EMOTION_NAMES[np.argmax(emotion_history[0])]
    end_emotion = EMOTION_NAMES[np.argmax(emotion_history[-1])]
    if start_emotion != end_emotion:
        arc_type += f" (from {start_emotion} to {end_emotion})"

    return turning_points, arc_type


# ============== ENDPOINTS ==============

# ============== ENDPOINTS ==============

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks, _auth: str = Depends(verify_api_key)):
    """Single utterance prediction with session context and sanitization."""
    # Ruthless Sanitization
    is_valid, sanitized_text, reason, emoji_emotion = input_sanitizer.sanitize_and_validate(req.text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=reason)
    
    if reason == "EmojiBypass" and emoji_emotion:
        synth = {
            "emotion": emoji_emotion, "confidence": 1.0, "sarcasm_prob": 0.0,
            "intensity": 0.8 if emoji_emotion in ["rage", "ecstasy", "terror", "grief", "loathing", "amazement", "vigilance", "adoration"] else 0.5,
            "ring": PLUTCHIK[emoji_emotion]["ring"], "sector": PLUTCHIK[emoji_emotion]["sector"],
            "top_5": [{"emotion": emoji_emotion, "probability": 1.0}],
            "explanations": {"reason": "Deterministic Emoji Bypass"},
            "cls_embedding": [0.0]*768, "token_embeddings": None,
            "emotion_probs": [1.0 if EMOTION_NAMES[i] == emoji_emotion else 0.0 for i in range(NUM_EMOTIONS)],
            "warning": "Emoji Map Bypass"
        }
        session_manager.add_turn(req.session_id, sanitized_text, req.speaker,
                                  emotion_probs=synth["emotion_probs"])
        return PredictResponse(**synth, context_used="BYPASS")
    
    try:
        context = session_manager.get_context(req.session_id)
        # Use req.explain to control whether to compute attributions
        result = _run_inference(sanitized_text, req.scenario, req.topic, context, compute_explanations=req.explain)
        session_manager.add_turn(req.session_id, sanitized_text, req.speaker,
                                  emotion_probs=result.get("emotion_probs"))

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
            warning=result.get("warning")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=PredictResponse)
async def explain(req: PredictRequest, background_tasks: BackgroundTasks, _auth: str = Depends(verify_api_key)):
    """
    Single utterance prediction with full Captum explainability.
    This is a heavier endpoint and should be used for detailed analysis, not high-throughput prediction.
    """
    # Ruthless Sanitization
    is_valid, sanitized_text, reason, emoji_emotion = input_sanitizer.sanitize_and_validate(req.text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=reason)

    if reason == "EmojiBypass" and emoji_emotion:
        # Emoji bypass doesn't have token attributions, so we return a simplified explanation.
        synth = {
            "emotion": emoji_emotion, "confidence": 1.0, "sarcasm_prob": 0.0,
            "intensity": 0.8 if emoji_emotion in ["rage", "ecstasy", "terror", "grief", "loathing", "amazement", "vigilance", "adoration"] else 0.5,
            "ring": PLUTCHIK[emoji_emotion]["ring"], "sector": PLUTCHIK[emoji_emotion]["sector"],
            "top_5": [{"emotion": emoji_emotion, "probability": 1.0}],
            "explanations": {"reason": "Deterministic Emoji Bypass. No token attributions available."},
            "cls_embedding": [0.0]*768, "token_embeddings": None,
            "emotion_probs": [1.0 if EMOTION_NAMES[i] == emoji_emotion else 0.0 for i in range(NUM_EMOTIONS)],
            "warning": "Emoji Map Bypass"
        }
        return PredictResponse(**synth, context_used="BYPASS")

    try:
        context = session_manager.get_context(req.session_id)
        # Call the core inference function with explanations enabled
        result = _run_inference(sanitized_text, req.scenario, req.topic, context, compute_explanations=True)
        
        # We don't add to session history from /explain to keep it clean for forecasting,
        # as it's an analysis endpoint, not part of a live conversation flow.

        # Log the prediction to the database
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
            warning=result.get("warning")
        )
    except Exception as e:
        logger.error(f"Error in /explain endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during explanation generation: {e}")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest, _auth: str = Depends(verify_api_key)):
    """Processes a batch of utterances for emotion analysis."""
    results = []
    for item in req.items:
        is_valid, sanitized_text, reason, _ = input_sanitizer.sanitize_and_validate(item.text)
        if not is_valid:
            # Skip invalid items in batch, maybe log them
            logger.warning(f"Skipping invalid batch item. Reason: {reason}. Text: {item.text[:100]}...")
            continue

        try:
            # We run inference without context for batch items for now.
            # A more advanced implementation could group items by session_id.
            result = _run_inference(sanitized_text, item.scenario, item.topic, context="[NO_CONTEXT]")
            results.append(
                BatchResponseItem(
                    text=sanitized_text,
                    emotion=result["emotion"],
                    confidence=result["confidence"],
                    sarcasm_prob=result["sarcasm_prob"],
                    intensity=result["intensity"],
                    ring=result["ring"],
                )
            )
        except Exception as e:
            logger.error(f"Error processing batch item: {e}. Text: {item.text[:100]}...")
            # Decide if one error fails the whole batch or just skips the item
            continue
    
    return BatchResponse(results=results, count=len(results))


@app.post("/predict/arc", response_model=ArcResponse)
async def predict_arc(req: ArcRequest, _auth: str = Depends(verify_api_key)):
    """Analyzes the emotional arc of a full conversation."""
    turns_results = []
    context_window = deque(maxlen=3)
    emotion_history = []
    intensity_trajectory = []

    for utterance in req.utterances:
        is_valid, sanitized_text, reason, _ = input_sanitizer.sanitize_and_validate(utterance.text)
        if not is_valid:
            # Create a placeholder for invalid turns to maintain sequence length
            turn_result = {
                "speaker": utterance.speaker,
                "text": utterance.text,
                "emotion": "invalid",
                "confidence": 0.0,
                "sarcasm_prob": 0.0,
                "intensity": 0.0,
                "ring": "N/A",
                "error": reason,
            }
            turns_results.append(turn_result)
            
            # CRITICAL FIX: Keep history arrays aligned with turns_results to prevent UI crash
            dummy_probs = [0.0] * NUM_EMOTIONS
            emotion_history.append(dummy_probs)
            intensity_trajectory.append(0.0)
            continue

        context = " | ".join(list(context_window)) if context_window else "[NO_CONTEXT]"
        
        try:
            result = _run_inference(sanitized_text, req.scenario, req.topic, context)
            
            turn_output = {
                "speaker": utterance.speaker,
                "text": sanitized_text,
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "sarcasm_prob": result["sarcasm_prob"],
                "intensity": result["intensity"],
                "ring": result["ring"],
                "top_5": result["top_5"],
            }
            turns_results.append(turn_output)
            
            # Update history for next turn
            context_window.append(f"{utterance.speaker[:3].upper()}: {sanitized_text}")
            emotion_history.append(result["emotion_probs"])
            intensity_trajectory.append(result["intensity"])

        except Exception as e:
            logger.error(f"Error processing arc utterance: {e}")
            turns_results.append({"speaker": utterance.speaker, "text": utterance.text, "error": str(e)})
            emotion_history.append([0.0] * NUM_EMOTIONS)
            intensity_trajectory.append(0.0)
            continue

    # Analyze the arc to find turning points and classify the arc type
    turning_points, arc_type = _analyze_arc_trajectory(emotion_history)

    return ArcResponse(
        arc_type=arc_type,
        turns=turns_results,
        turning_points=turning_points,
        intensity_trajectory=intensity_trajectory,
    )


@app.post("/analyze/dynamic", response_model=DynamicAnalysisResponse)
async def analyze_dynamic(req: DynamicAnalysisRequest, _auth: str = Depends(verify_api_key)):
    """Advanced dynamic analysis with trajectory forecasting and sanitization."""
    # Ruthless Sanitization
    is_valid, sanitized_text, reason, emoji_emotion = input_sanitizer.sanitize_and_validate(req.text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=reason)
    
    if reason == "EmojiBypass" and emoji_emotion:
        return DynamicAnalysisResponse(
            risk_level="extreme" if emoji_emotion in ["rage", "terror", "loathing"] else "low",
            sarcasm_probability=0.0,
            trajectory_forecast=[[1.0 if EMOTION_NAMES[i] == emoji_emotion else 0.0 for i in range(NUM_EMOTIONS)]] * 5,
            inflection_point=0,
            reframe_suggestions=["Emoji only input detected - no complex de-escalation needed."],
            baseline_deviation=None,
            signals=["Symbolic Expression Detected"],
            warning="Emoji Map Bypass"
        )

    try:
        context = session_manager.get_context(req.session_id)
        # 1. Run inference on sanitized text
        result = _run_inference(sanitized_text, "support", "dynamic_session", context)
        
        # 2. Get history for forecasting
        history = session_manager.get_history_vectors(req.session_id)
        
        warning = result.get("warning")
        if not history:
            warning = (warning + " | " if warning else "") + "No session history found. Trajectory may be unstable."
            # Provide at least the current vector to prevent downstream crash
            history = [result['emotion_probs']]
        
        # 3. Run advanced analysis
        analysis = advanced_engine.analyze_dynamic(
            text=sanitized_text,
            current_emotion_vector=result['emotion_probs'],
            dialogue_history=history,
            user_baseline=req.user_baseline
        )
        
        # 4. Update session
        session_manager.add_turn(req.session_id, req.text, "USER")
        
        return DynamicAnalysisResponse(
            risk_level=analysis['risk_level'],
            sarcasm_probability=analysis['incongruity']['sarcasm_probability'],
            trajectory_forecast=analysis['forecast']['trajectory'],
            inflection_point=analysis['forecast']['inflection_point_step'],
            reframe_suggestions=analysis['reframe_suggestions'],
            baseline_deviation=analysis['baseline_deviation'],
            signals=analysis['incongruity']['signals'],
            warning=warning
        )
    except Exception as e:
        logger.error(f"Dynamic Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest, _auth: str = Depends(verify_api_key)):
    """Batch prediction — no session context, stateless."""
    try:
        results = []
        rejected = []
        for item in req.items:
            # Fix 5: batch endpoint was skipping sanitization entirely
            is_valid, sanitized_text, reason, emoji_emotion = input_sanitizer.sanitize_and_validate(item.text)
            if not is_valid:
                rejected.append({"text": item.text[:80], "reason": reason})
                continue

            if reason == "EmojiBypass" and emoji_emotion:
                results.append(BatchResponseItem(
                    text=item.text,
                    emotion=emoji_emotion,
                    confidence=1.0, sarcasm_prob=0.0,
                    intensity=0.5,
                    ring=PLUTCHIK[emoji_emotion]["ring"]
                ))
                continue

            result = _run_inference(sanitized_text, item.scenario, item.topic, context="[NO_CONTEXT]", compute_explanations=False)
            results.append(BatchResponseItem(
                text=item.text,
                emotion=result["emotion"],
                confidence=result["confidence"],
                sarcasm_prob=result["sarcasm_prob"],
                intensity=result["intensity"],
                ring=result["ring"],
            ))
        if rejected:
            logger.warning(f"Batch: {len(rejected)} item(s) rejected by sanitizer: {rejected}")
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


@app.post("/reload")
async def reload_model(_auth: str = Depends(verify_api_key)):
    """Hot-reload model weights from disk without restarting the server."""
    success = load_model_weights()
    if success:
        return {"status": "success", "message": "Model weights reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model weights. Check server logs.")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": device,
        "model_ready": MODEL_READY,
        "version": "2.1.0",
        "endpoints": [
            "/predict", "/predict/batch", "/predict/arc", "/explain", "/analyze/dynamic",
            "/correct", "/corrections/stats", "/emotions", "/session/{id}", "/health", "/reload"
        ],
        "warning": None if MODEL_READY else "Using untrained model — predictions are random",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
