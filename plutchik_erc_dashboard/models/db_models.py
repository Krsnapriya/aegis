from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import sys
from pathlib import Path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))
from database import Base

class DB_Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    emotion = Column(String)
    confidence = Column(Float)
    sarcasm_prob = Column(Float)
    intensity = Column(Float)
    ring = Column(String)
    sector = Column(String)
    scenario = Column(String)
    topic = Column(String)
    speaker = Column(String)
    context_used = Column(String)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to corrections
    corrections = relationship("DB_Correction", back_populates="prediction")

class DB_Correction(Base):
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    text = Column(String)
    predicted_emotion = Column(String)
    corrected_emotion = Column(String)
    predicted_confidence = Column(Float)
    status = Column(String, default="pending_review") # pending_review, reviewed, ingested
    annotator_notes = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("DB_Prediction", back_populates="corrections")

class DB_DialogueTurn(Base):
    __tablename__ = "dialogue_turns"

    id = Column(Integer, primary_key=True, index=True)
    dialogue_id = Column(String, index=True) # UUID for a full conversation
    turn_index = Column(Integer)
    speaker = Column(String)
    text = Column(String)
    emotion = Column(String)
    confidence = Column(Float)
    intensity = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
