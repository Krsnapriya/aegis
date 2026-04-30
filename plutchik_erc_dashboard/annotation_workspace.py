"""
Plutchik ERC v2.1 — Annotation Workspace (Surface 5)
Collaborative HITL labeling environment for emotion corrections.

Launch: streamlit run annotation_workspace.py
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
from sqlalchemy.orm import Session
from sqlalchemy import func

# Page config
st.set_page_config(
    page_title="Plutchik Annotation Workspace",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== DATABASE INTEGRATION ==============
PROJECT_DIR = Path(__file__).parent
import sys
sys.path.insert(0, str(PROJECT_DIR))

from database import SessionLocal, get_db
from models.db_models import DB_Correction, DB_Prediction
from utils.constants import PLUTCHIK, EMOTION_NAMES, RING_INTENSITY

def get_db_session():
    """Returns a new database session."""
    return SessionLocal()

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;500;700;900&display=swap');

    :root {
        --bg-color: #0d1117;
        --surface-card: rgba(22, 27, 34, 0.8);
        --surface-border: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-green: #3fb950;
        --accent-blue: #58a6ff;
        --accent-red: #ff7b72;
        --accent-purple: #a371f7;
        --accent-orange: #f0883e;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    .correction-card {
        background: var(--surface-card);
        border: 1px solid var(--surface-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }

    .correction-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }

    .status-pending { color: var(--accent-orange); font-weight: 600; }
    .status-reviewed { color: var(--accent-green); font-weight: 600; }

    .stat-card {
        background: var(--surface-card);
        border: 1px solid var(--surface-border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .stat-number {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent-blue);
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    .emotion-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }

    .ring-intense { background: rgba(255,123,114,0.2); color: #ff7b72; }
    .ring-primary { background: rgba(88,166,255,0.2); color: #58a6ff; }
    .ring-mild { background: rgba(163,113,247,0.2); color: #a371f7; }
    .ring-dyadic { background: rgba(63,185,80,0.2); color: #3fb950; }
</style>
""", unsafe_allow_html=True)


# ============== UI ==============
st.markdown("<h1 style='text-align: center;'>🏷️ Annotation Workspace</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;'>Collaborative HITL labeling for Plutchik ERC</p>", unsafe_allow_html=True)

with get_db_session() as db:
    # Load data from DB
    total_corrections = db.query(DB_Correction).count()
    pending_list = db.query(DB_Correction).filter(DB_Correction.status == "pending_review").order_by(DB_Correction.timestamp.desc()).all()
    reviewed_list = db.query(DB_Correction).filter(DB_Correction.status == "reviewed").all()
    
    top_emotion_res = db.query(DB_Correction.corrected_emotion, func.count(DB_Correction.corrected_emotion)).group_by(DB_Correction.corrected_emotion).order_by(func.count(DB_Correction.corrected_emotion).desc()).first()
    top_emotion = top_emotion_res[0] if top_emotion_res else "—"

# ============== STATS HEADER ==============
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='stat-card'><div class='stat-number'>{total_corrections}</div><div class='stat-label'>Total Corrections</div></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='stat-card'><div class='stat-number' style='color: var(--accent-orange);'>{len(pending_list)}</div><div class='stat-label'>Pending Review</div></div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='stat-card'><div class='stat-number' style='color: var(--accent-green);'>{len(reviewed_list)}</div><div class='stat-label'>Reviewed</div></div>", unsafe_allow_html=True)

with col4:
    st.markdown(f"<div class='stat-card'><div class='stat-number' style='color: var(--accent-purple); font-size: 1.5rem;'>{top_emotion.title()}</div><div class='stat-label'>Most Corrected To</div></div>", unsafe_allow_html=True)


# ============== TABS ==============
tab1, tab2, tab3 = st.tabs(["📝 Submit Correction", "📋 Review Queue", "📊 Analytics"])

# --- Tab 1: Submit New Correction ---
with tab1:
    st.markdown("### Submit a New Correction")
    st.caption("Use this when you notice the model predicted the wrong emotion.")

    with st.form("correction_form"):
        text_input = st.text_area("Utterance Text", placeholder="Enter the text that was misclassified...")
        col_a, col_b = st.columns(2)
        with col_a:
            predicted = st.selectbox("Model's Prediction", EMOTION_NAMES, index=0)
        with col_b:
            corrected = st.selectbox("Correct Emotion", EMOTION_NAMES, index=1)

        col_c, col_d = st.columns(2)
        with col_c:
            confidence = st.slider("Model's Confidence", 0.0, 1.0, 0.5, 0.01)
        with col_d:
            notes = st.text_area("Notes (optional)", placeholder="Why is the model wrong here?")

        submitted = st.form_submit_button("Submit Correction", type="primary")

        if submitted and text_input.strip():
            with get_db_session() as db:
                new_corr = DB_Correction(
                    text=text_input,
                    predicted_emotion=predicted,
                    corrected_emotion=corrected,
                    predicted_confidence=confidence,
                    status="pending_review",
                    annotator_notes=notes
                )
                db.add(new_corr)
                db.commit()
            st.success("✓ Correction submitted to database.")
            st.rerun()

# --- Tab 2: Review Queue ---
with tab2:
    st.markdown("### Pending Review")

    if not pending_list:
        st.info("🎉 No corrections pending review! The queue is empty.")
    else:
        for corr in pending_list:
            ring_pred = PLUTCHIK.get(corr.predicted_emotion, {}).get("ring", "primary")
            ring_corr = PLUTCHIK.get(corr.corrected_emotion, {}).get("ring", "primary")

            st.markdown(f"""
            <div class='correction-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span class='status-pending'>⏳ PENDING</span>
                    <span style='color: var(--text-secondary); font-size: 0.8rem;'>ID: {corr.id}</span>
                </div>
                <p style='margin: 1rem 0; font-size: 1.05rem; line-height: 1.6;'>"{corr.text}"</p>
                <div style='display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;'>
                    <span class='emotion-tag ring-{ring_pred}'>❌ {corr.predicted_emotion.title()}</span>
                    <span style='color: var(--text-secondary);'>→</span>
                    <span class='emotion-tag ring-{ring_corr}'>✓ {corr.corrected_emotion.title()}</span>
                    <span style='color: var(--text-secondary); font-size: 0.8rem; margin-left: auto;'>
                        conf: {corr.predicted_confidence:.0%} · {corr.timestamp.strftime('%Y-%m-%d')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_approve, col_reject, col_spacer = st.columns([1, 1, 4])
            with col_approve:
                if st.button(f"✅ Approve", key=f"approve_{corr.id}"):
                    with get_db_session() as db:
                        db.merge(corr)
                        corr.status = "reviewed"
                        db.commit()
                    st.rerun()
            with col_reject:
                if st.button(f"❌ Reject", key=f"reject_{corr.id}"):
                    with get_db_session() as db:
                        db.delete(db.merge(corr))
                        db.commit()
                    st.rerun()

# --- Tab 3: Analytics ---
with tab3:
    st.markdown("### Correction Analytics")

    if total_corrections == 0:
        st.info("No corrections yet.")
    else:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Model Errors (Corrected FROM)")
            with get_db_session() as db:
                err_stats = db.query(DB_Correction.predicted_emotion, func.count(DB_Correction.predicted_emotion)).group_by(DB_Correction.predicted_emotion).order_by(func.count(DB_Correction.predicted_emotion).desc()).limit(10).all()
            for emotion, count in err_stats:
                bar_width = int((count / err_stats[0][1]) * 100)
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin: 0.3rem 0;'>
                    <span style='width: 120px; font-size: 0.85rem;'>{emotion.title()}</span>
                    <div style='flex: 1; background: var(--surface-border); border-radius: 4px; height: 20px;'><div style='width: {bar_width}%; background: var(--accent-red); height: 100%; border-radius: 4px;'></div></div>
                    <span style='width: 30px; text-align: right; font-size: 0.85rem; margin-left: 0.5rem;'>{count}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### True Labels (Corrected TO)")
            with get_db_session() as db:
                target_stats = db.query(DB_Correction.corrected_emotion, func.count(DB_Correction.corrected_emotion)).group_by(DB_Correction.corrected_emotion).order_by(func.count(DB_Correction.corrected_emotion).desc()).limit(10).all()
            for emotion, count in target_stats:
                bar_width = int((count / target_stats[0][1]) * 100)
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin: 0.3rem 0;'>
                    <span style='width: 120px; font-size: 0.85rem;'>{emotion.title()}</span>
                    <div style='flex: 1; background: var(--surface-border); border-radius: 4px; height: 20px;'><div style='width: {bar_width}%; background: var(--accent-green); height: 100%; border-radius: 4px;'></div></div>
                    <span style='width: 30px; text-align: right; font-size: 0.85rem; margin-left: 0.5rem;'>{count}</span>
                </div>
                """, unsafe_allow_html=True)

# ============== SIDEBAR ==============
st.sidebar.markdown("### 🔄 Retrain Readiness")
reviewed_count = len(reviewed_list)
retrain_threshold = 50
st.sidebar.progress(min(reviewed_count / retrain_threshold, 1.0))
st.sidebar.caption(f"{reviewed_count}/{retrain_threshold} reviewed corrections")

if st.sidebar.button("📥 Export Reviewed (JSON)"):
    data = [{"text": c.text, "label": c.corrected_emotion} for c in reviewed_list]
    st.sidebar.download_button("Download", json.dumps(data, indent=2), "reviewed_corrections.json")

st.sidebar.markdown("---")
st.sidebar.markdown(f"<div style='text-align: center; color: #6c757d; font-size: 0.7rem;'>Plutchik ERC Workspace v2.2 (SQL)<br>© 2026 Plutchik ERC Project</div>", unsafe_allow_html=True)
