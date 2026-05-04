"""
Streamlit Dashboard for Plutchik Emotion Recognition with Explainability.
"""

from pathlib import Path
import sys

_pkg = Path(__file__).resolve().parent
_repo = _pkg.parent
_core = _repo / "core"
for _path in (_core, _pkg):
    _ps = str(_path)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import requests
import html
from typing import Dict, List

from models.multitask_emotion_model import PluTchikMultiTaskModel
from utils.preprocessing import ERCPreprocessor
from utils.explainability import ExplainabilityEngine
from utils.explainability_v2 import CaptumExplainer
from utils.trainer import PluTchikTrainer
from utils.llm_inference import NemotronClient
from utils.constants import PLUTCHIK, PRIMARY_EMOTIONS, EMOTION_NAMES, NUM_EMOTIONS
from dotenv import load_dotenv

# Load environment variables from .env file for local development.
# In a production environment, these should be set as actual environment variables.
load_dotenv(dotenv_path=_repo / ".env")


# ============== PAGE CONFIG ==============
st.set_page_config(page_title="Plutchik ERC", page_icon="🎭", layout="wide")

# ============== SESSION STATE INIT ==============
if "history" not in st.session_state:
    st.session_state.history = []
if "history_buffer" not in st.session_state:
    st.session_state.history_buffer = []
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============== CUSTOM CSS: PREMIUM AESTHETICS ==============
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;500;700;900&family=Fira+Code:wght@400;500&display=swap');
    
    :root {
        --bg-color: #0b0e14;
        --surface-color: rgba(22, 27, 34, 0.7);
        --surface-border: rgba(48, 54, 61, 0.6);
        --accent-primary: #58a6ff;
        --accent-secondary: #bc8cff;
        --accent-gradient: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --glass-bg: rgba(13, 17, 23, 0.8);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background: radial-gradient(circle at top right, #161b22, #0b0e14);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism utility */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-primary) !important;
    }

    h1 {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem !important;
        margin-bottom: 0rem !important;
        filter: drop-shadow(0 0 15px rgba(88, 166, 255, 0.3));
    }

    /* Subtitle animation */
    .subtitle {
        color: var(--text-secondary);
        font-size: 1.25rem;
        font-weight: 400;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: -0.5rem;
        margin-bottom: 3rem;
        opacity: 0.8;
    }

    /* Clean up text areas */
    .stTextArea textarea {
        background-color: #0d1117 !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        font-size: 1.1rem !important;
        padding: 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
        transform: translateY(-2px);
    }

    /* Primary Button */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.25) !important;
        width: 100% !important;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(8px);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        border-color: var(--accent-primary);
        background: rgba(88, 166, 255, 0.05);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0e14;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding: 10px;
        border-bottom: 2px solid var(--surface-border);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 2rem;
        border-radius: 12px 12px 0 0;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(88, 166, 255, 0.1) !important;
        border-bottom: 3px solid var(--accent-primary) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid var(--surface-border);
    }
    </style>
    """, unsafe_allow_html=True)

# ============== UI HEADER ==============
st.markdown("<h1 style='text-align: center;'>PLUTCHIK AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='text-align: center;'>Beyond Words: Decoding the Emotional DNA</p>", unsafe_allow_html=True)


# ============== API INTEGRATION (Thin Client) ==============
API_BASE = os.getenv("PLUTCHIK_API_URL", "http://localhost:8000")
API_KEY = os.getenv("PLUTCHIK_API_KEY")

def call_api(endpoint: str, payload: dict, use_auth: bool = True):
    headers = {"Content-Type": "application/json"}
    if use_auth:
        headers["X-API-Key"] = API_KEY
    
    try:
        response = requests.post(f"{API_BASE}/{endpoint}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# Load utilities
preprocessor = ERCPreprocessor(PLUTCHIK)
llm_client = NemotronClient()

# Load centroids for embedding similarity (lightweight)
@st.cache_resource
def load_centroids():
    model_dir = Path(__file__).parent / "my_plutchik_model"
    centroids_path = model_dir / "emotion_centroids.pkl"
    if centroids_path.exists():
        with open(centroids_path, "rb") as f:
            return pickle.load(f)
    return {}

emotion_centroids = load_centroids()


# ============== SIDEBAR CONFIGURATION ==============
with st.sidebar:
    st.markdown("### ⚙️ Engine Control")
    analysis_mode = st.radio(
        "Analysis Protocol", 
        ["Single Utterance", "Conversation Arc", "Comparative Analysis", "Dynamic Intelligence", "Batch File Upload"],
        help="Choose the scale and type of emotional analysis."
    )
    model_type = st.radio("Inference Core", ["Local RoBERTa", "Nemotron-3 (LLM)", "Compare Both Models"])

    st.markdown("---")
    st.markdown("### 📍 Context Matrix")
    
    scenario = st.selectbox(
        "Scenario Environment",
        ["workplace", "friendship", "family", "romance", "support", "academic", 
         "conflict", "casual", "social", "travel", "technology", "creative", "wellbeing", "community"]
    )
    
    topic = st.text_input("Operational Topic", value="general")
    speaker = st.text_input("Source Persona", value="USER")

    st.markdown("---")
    use_history = st.checkbox("Persistent Context", value=True)
    if not use_history:
        prev_turns_manual = st.text_area("Manual Context Buffer", placeholder="Turn 1 | Turn 2...")
    else:
        st.caption("Using session history for context-aware inference.")

    use_captum_explain = False
    if analysis_mode == "Single Utterance":
        use_captum_explain = st.checkbox(
            "Full explainability (Captum on full context window; slower)",
            value=True,
            help="Calls POST /explain so token IG matches [CONTEXT]…[CURRENT]… input seen by the model.",
        )
    batch_max_rows = 200
    if analysis_mode == "Batch File Upload":
        batch_max_rows = st.number_input("Max CSV rows to score", min_value=10, max_value=2000, value=200, step=10)

# ============== MAIN UI: INPUT SECTION ==============
input_container = st.container()
with input_container:
    if analysis_mode == "Single Utterance":
        user_text = st.text_area(
            label="Input Signal",
            label_visibility="collapsed",
            placeholder="Transmit message for emotional decoding...",
            height=160
        )
    elif analysis_mode == "Conversation Arc":
        user_text = st.text_area(
            label="Dialogue Data",
            label_visibility="collapsed",
            placeholder="Enter dialogue stream (format SPEAKER: TEXT)\n\nExample:\nUSER: This is unacceptable!\nAGENT: I'm so sorry you're feeling this way.",
            height=280
        )
    elif analysis_mode == "Comparative Analysis":
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### Stream A")
            user_text_1 = st.text_area("A", label_visibility="collapsed", placeholder="USER: Hello\nAGENT: Hi", height=200)
        with col_v2:
            st.markdown("#### Stream B")
            user_text_2 = st.text_area("B", label_visibility="collapsed", placeholder="USER: Hello\nAGENT: GO AWAY", height=200)
        user_text = user_text_1 # For button validation
    else:
        uploaded_file = st.file_uploader("Upload Signal Batch (CSV)", type=["csv"])
        user_text = "FILE_UPLOADED" if uploaded_file else ""


# ============== INFERENCE ==============
st.markdown("<br>", unsafe_allow_html=True)
if st.button("✨ Run Analysis", key="predict_btn", use_container_width=True):
    if not user_text.strip():
        st.warning("Please provide input for analysis.")
    else:
        if analysis_mode == "Comparative Analysis":
            with st.spinner("🧬 Comparing trajectories..."):
                results = []
                for i, raw in enumerate([user_text_1, user_text_2]):
                    utts = []
                    for line in raw.split("\n"):
                        if ":" in line:
                            spk, txt = line.split(":", 1)
                            utts.append({"speaker": spk.strip(), "text": txt.strip()})
                    if utts:
                        res = call_api("predict/arc", {"utterances": utts, "scenario": scenario, "topic": topic})
                        if res: results.append(res)
                st.session_state.compare_results = results
                st.session_state.arc_prediction = None
                st.session_state.prediction = None
        elif analysis_mode == "Batch File Upload":
            with st.spinner("📦 Processing batch file..."):
                df = pd.read_csv(uploaded_file)
                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    texts = df["text"].astype(str).tolist()[: int(batch_max_rows)]
                    batch_req = {"items": [{"text": t, "scenario": scenario, "topic": topic} for t in texts]}
                    res = call_api("predict/batch", batch_req)
                    if res:
                        st.session_state.batch_results = pd.DataFrame(res["results"])
                        st.session_state.prediction = None
        elif analysis_mode == "Conversation Arc":
            # Move existing logic here
            with st.spinner("🧬 Analyzing conversation trajectory..."):
                utterances = []
                for line in user_text.split("\n"):
                    if ":" in line:
                        spk, txt = line.split(":", 1)
                        utterances.append({"speaker": spk.strip(), "text": txt.strip()})
                if utterances:
                    arc_res = call_api("predict/arc", {"utterances": utterances, "scenario": scenario, "topic": topic})
                    if arc_res:
                        st.session_state.arc_prediction = arc_res
                        st.session_state.prediction = None
        elif analysis_mode == "Dynamic Intelligence":
            with st.spinner("🧠 Initializing Advanced Dynamic Engine..."):
                payload = {
                    "text": user_text,
                    "session_id": "dashboard_session",
                    "user_baseline": None # Could be expanded in future
                }
                dynamic_res = call_api("analyze/dynamic", payload, use_auth=True)
                if dynamic_res:
                    st.session_state.dynamic_analysis = dynamic_res
                    st.session_state.prediction = None
                    st.session_state.arc_prediction = None
        else:
            with st.spinner(f"🔮 Analyzing via {model_type}..."):
                # 1. Prepare Context
                if use_history and st.session_state.history_buffer:
                    context_str = " | ".join(st.session_state.history_buffer[-3:])
                elif not use_history and prev_turns_manual:
                    context_str = prev_turns_manual
                else:
                    context_str = "[NO_CONTEXT]"
            
            if model_type == "Local RoBERTa":
                # API-based Inference (Thin Client)
                payload = {
                    "text": user_text,
                    "session_id": "dashboard_session",
                    "speaker": speaker,
                    "scenario": scenario,
                    "topic": topic
                }
                ep = "explain" if use_captum_explain else "predict"
                result = call_api(ep, payload)
                
                if result:
                    expl = result.get("explanations") or {}
                    st.session_state.prediction = {
                        "text": user_text,
                        "emotion": result["emotion"],
                        "emotion_confidence": result["confidence"],
                        "emotion_probs": np.array(result.get("emotion_probs", [])),
                        "emotion_names": sorted(PLUTCHIK.keys()),
                        "sarcasm_confidence": result["sarcasm_prob"],
                        "intensity": result["intensity"],
                        "attribution_data": expl.get("token_attributions", []),
                        "context_span_top": expl.get("context_span_top", []),
                        "current_span_top": expl.get("current_span_top", []),
                        "assessment": result.get("assessment"),
                        "context_used": result["context_used"],
                        "model_type": model_type,
                        "embedding_info": {
                            "cls_embedding": np.array(result.get("cls_embedding", [])),
                            "all_token_embeddings": np.array(result.get("token_embeddings", []))
                            if result.get("token_embeddings") is not None else None
                        }
                    }
            elif model_type == "Nemotron-3 (LLM)":
                # LLM Inference Logic
                llm_res = llm_client.predict_emotion(user_text, scenario, topic, context_str)
                
                if "error" in llm_res:
                    st.error(f"❌ LLM Error: {llm_res['error']}")
                else:
                    # Map LLM results to dashboard structure
                    emotion_names = sorted(PLUTCHIK.keys())
                    
                    # Create dummy probs for visualization
                    dummy_probs = np.zeros(len(emotion_names))
                    predicted_emotion = llm_res.get("emotion", "neutral")
                    if predicted_emotion in emotion_names:
                        idx = emotion_names.index(predicted_emotion)
                        dummy_probs[idx] = 1.0
                    
                    st.session_state.prediction = {
                        "text": user_text,
                        "emotion": predicted_emotion,
                        "emotion_confidence": 1.0, # LLM is "confident" in its single choice
                        "emotion_probs": dummy_probs,
                        "emotion_names": emotion_names,
                        "sarcasm_confidence": float(llm_res.get("sarcasm_confidence", 0.0)),
                        "intensity": float(llm_res.get("intensity", 0.5)),
                        "reasoning": str(llm_res.get("reasoning", "No reasoning provided by LLM.")),
                        "context_used": context_str,
                        "model_type": model_type,
                        "attribution_data": None # LLM doesn't provide attribution (yet)
                    }
                
            elif model_type == "Compare Both Models":
                # 1. Local Run via API
                payload = {
                    "text": user_text,
                    "session_id": "dashboard_session",
                    "speaker": speaker,
                    "scenario": scenario,
                    "topic": topic
                }
                
                result = call_api("predict", payload)
                
                if result:
                    local_pred = {
                        "text": user_text,
                        "emotion": result["emotion"],
                        "emotion_confidence": result["confidence"],
                        "emotion_probs": np.array(result.get("emotion_probs", [])),
                        "emotion_names": sorted(PLUTCHIK.keys()),
                        "sarcasm_confidence": result["sarcasm_prob"],
                        "intensity": result["intensity"],
                        "attribution_data": result.get("explanations", {}).get("token_attributions", []) if result.get("explanations") else [],
                        "context_used": result["context_used"],
                        "model_type": "Local RoBERTa",
                        "embedding_info": {
                            "cls_embedding": np.array(result.get("cls_embedding", []))
                        }
                    }
                else:
                    local_pred = None
                
                # 2. LLM Run
                llm_res = llm_client.predict_emotion(user_text, scenario, topic, context_str)
                
                if "error" in llm_res:
                    st.error(f"❌ LLM Error: {llm_res['error']}")
                    llm_pred = None
                else:
                    emotion_names = sorted(PLUTCHIK.keys())
                    dummy_probs = np.zeros(len(emotion_names))
                    predicted_emotion = llm_res.get("emotion", "neutral")
                    if predicted_emotion in emotion_names:
                        idx = emotion_names.index(predicted_emotion)
                        dummy_probs[idx] = 1.0
                    
                    llm_pred = {
                        "text": user_text,
                        "emotion": predicted_emotion,
                        "emotion_confidence": 1.0,
                        "emotion_probs": dummy_probs,
                        "emotion_names": emotion_names,
                        "sarcasm_confidence": float(llm_res.get("sarcasm_confidence", 0.0)),
                        "intensity": float(llm_res.get("intensity", 0.5)),
                        "reasoning": str(llm_res.get("reasoning", "No reasoning provided by LLM.")),
                        "context_used": context_str,
                        "model_type": "Nemotron-3 (LLM)",
                        "attribution_data": None
                    }
                
                st.session_state.comparison = {
                    "local": local_pred,
                    "llm": llm_pred
                }
                st.session_state.prediction = local_pred # Set default for visualizations
            
            # Update history buffer
            st.session_state.history_buffer.append(f"{speaker[:3].upper()}: {user_text}")
            st.success(f"✓ {model_type} analysis complete")


# ============== DISPLAY RESULTS ==============
if "prediction" in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "comparison" in st.session_state and model_type == "Compare Both Models":
        st.markdown("### ⚖️ Model Comparison")
        c1, c2 = st.columns(2)
        local = st.session_state.comparison["local"]
        llm = st.session_state.comparison["llm"]
        
        with c1:
            st.info("**🤖 Local RoBERTa**")
            st.metric("Emotion", local["emotion"].title(), f"{local['emotion_confidence']:.1%}")
            st.progress(local["sarcasm_confidence"], f"Sarcasm: {local['sarcasm_confidence']:.1%}")
        
        with c2:
            if llm:
                st.success("**🌟 Nemotron-3 (LLM)**")
                st.metric("Emotion", llm["emotion"].title(), "100%")
                st.progress(llm["sarcasm_confidence"], f"Sarcasm: {llm['sarcasm_confidence']:.1%}")
                with st.expander("View LLM Reasoning"):
                    st.write(llm["reasoning"])
            else:
                st.error("Nemotron-3 failed to return a result.")
        st.divider()

# ============== RESULTS DISPLAY ==============
if st.session_state.get("batch_results") is not None:
    st.markdown("### 📦 Batch Analysis Results")
    br = st.session_state.batch_results
    st.dataframe(br, use_container_width=True)
    csv = br.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results (CSV)", csv, "plutchik_batch_results.csv", "text/csv")

    st.markdown("#### Batch statistical profile")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows scored", len(br))
    if "emotion" in br.columns:
        m2.metric("Unique emotions", br["emotion"].nunique())
    if "sarcasm_prob" in br.columns:
        rate = float((br["sarcasm_prob"] >= 0.5).mean())
        m3.metric("Sarcasm rate (p≥0.5)", f"{rate:.1%}")
        m4.metric("Mean sarcasm p", f"{float(br['sarcasm_prob'].mean()):.1%}")
    if "emotion" in br.columns:
        vc = br["emotion"].value_counts().head(8)
        fig_em_dist = go.Figure(
            data=go.Bar(x=vc.values, y=[e.title() for e in vc.index], orientation="h", marker_color="#58a6ff")
        )
        fig_em_dist.update_layout(
            title="Top emotions (count)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"),
            height=320, margin=dict(l=120),
        )
        st.plotly_chart(fig_em_dist, use_container_width=True)
    if "ring" in br.columns:
        ring_vc = br["ring"].value_counts()
        fig_ring = px.pie(values=ring_vc.values, names=[r.title() for r in ring_vc.index], title="Ring distribution")
        fig_ring.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"))
        st.plotly_chart(fig_ring, use_container_width=True)
    if "sarcasm_prob" in br.columns:
        st.caption("Mean sarcasm probability: {:.1%}".format(float(br["sarcasm_prob"].mean())))

elif st.session_state.get("compare_results"):
    st.markdown("### 🧬 Comparative Trajectory Analysis")
    fig_comp = go.Figure()
    colors = ["#58a6ff", "#ff7b72"]
    for i, res in enumerate(st.session_state.compare_results):
        fig_comp.add_trace(go.Scatter(
            x=list(range(len(res["intensity_trajectory"]))),
            y=res["intensity_trajectory"],
            mode='lines+markers',
            name=f"Conversation {chr(65+i)}",
            line=dict(color=colors[i], width=3)
        ))
    fig_comp.update_layout(
        title="Divergent Emotional Arcs (intensity proxy)",
        xaxis_title="Turn", yaxis_title="Intensity",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e6edf3')
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Emotion label distribution (per conversation)")
    ec1, ec2 = st.columns(2)
    for i, res in enumerate(st.session_state.compare_results):
        turns = res.get("turns") or []
        col = ec1 if i == 0 else ec2
        with col:
            if turns:
                em_df = pd.DataFrame(turns)["emotion"].value_counts()
                fig_em = go.Figure(
                    data=go.Bar(
                        x=em_df.values,
                        y=[e.title() for e in em_df.index],
                        orientation="h",
                        marker_color=colors[i],
                    )
                )
                fig_em.update_layout(
                    title=f"Conversation {chr(65+i)}",
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e6edf3"),
                )
                st.plotly_chart(fig_em, use_container_width=True)

    cols = st.columns(max(1, len(st.session_state.compare_results)))
    for idx, res in enumerate(st.session_state.compare_results):
        with cols[idx]:
            st.markdown(f"**Conversation {chr(65+idx)}** — {res.get('arc_type', '').title()}")
            tdf = pd.DataFrame(res.get("turns", []))
            if not tdf.empty:
                want = [c for c in ["turn", "speaker", "emotion", "confidence", "sarcasm_prob", "intensity", "ring"] if c in tdf.columns]
                tdf = tdf[want] if want else tdf
            st.dataframe(tdf, use_container_width=True, height=280)

elif st.session_state.get("arc_prediction"):
    arc = st.session_state.arc_prediction
    
    st.markdown(f"### 📈 Conversation Arc: <span style='color: var(--accent-primary);'>{arc['arc_type'].upper()}</span>", unsafe_allow_html=True)
    
    # 1. Timeline Chart
    turns = arc["turns"]
    df_arc = pd.DataFrame(turns)
    df_arc["intensity_val"] = arc["intensity_trajectory"]
    
    fig_timeline = go.Figure()
    
    # Line for trajectory
    fig_timeline.add_trace(go.Scatter(
        x=df_arc["turn"],
        y=df_arc["intensity_val"],
        mode='lines+markers',
        name='Emotional Intensity',
        line=dict(color='#58a6ff', width=3),
        marker=dict(size=12, color='#58a6ff', symbol='circle'),
        text=[f"{t['speaker']}: {t['emotion']}" for t in turns],
        hoverinfo='text+y'
    ))
    
    # Highlight inflection points
    for tp in arc.get("turning_points", []):
        tp_color = "#ff7b72" if tp["type"] == "intensity_shift" else "#a371f7"
        fig_timeline.add_vline(
            x=tp["turn"], 
            line_dash="dash", 
            line_color=tp_color,
            annotation_text=f"Shift: {tp['type'].replace('_', ' ')}",
            annotation_position="top left"
        )
    
    fig_timeline.update_layout(
        title="Dialogue Emotion Trajectory",
        xaxis_title="Turn Number",
        yaxis_title="Intensity (Mild → Intense)",
        yaxis=dict(range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8], ticktext=["Mild", "Primary", "Intense", "Extreme"]),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6edf3'),
        xaxis=dict(showgrid=True, gridcolor='#30363d', zerolinecolor='#30363d'),
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # 2. Detailed Turn Table
    st.markdown("#### Turn-by-Turn Analysis")
    for t in turns:
        with st.expander(f"Turn {t['turn']}: {t['speaker']} — {t['emotion'].title()}"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Confidence", f"{t['confidence']:.1%}")
            col_b.metric("Intensity", f"{t['intensity']:.2f}")
            col_c.metric("Sarcasm", "Yes" if t['sarcasm_prob'] > 0.5 else "No")
            st.write(f"**Text:** {t['text']}")
    
    st.divider()

elif st.session_state.get("dynamic_analysis"):
    dyn = st.session_state.dynamic_analysis
    
    st.markdown(f"### 🧠 Dynamic Emotional Intelligence: <span style='color: var(--accent-primary);'>{dyn['risk_level'].upper()} RISK</span>", unsafe_allow_html=True)
    
    # 1. Summary Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Sarcasm Probability", f"{dyn['sarcasm_probability']:.1%}")
    c2.metric("Risk Level", dyn["risk_level"].title())
    c3.metric("Inflection Point", f"Step {dyn['inflection_point']}")
    
    # 2. Trajectory Forecast (Neural ODE)
    st.markdown("#### 📉 Neural ODE Trajectory Forecast")
    st.caption("⚠️ Simulation mode: forecaster weights are untrained (Phase 8 Ext.1 pending). Shape is illustrative.")
    traj = np.array(dyn["trajectory_forecast"])  # expected [Steps, 32]

    if traj.ndim != 2 or traj.shape[1] == 0:
        st.warning("Trajectory data has unexpected shape — skipping chart.")
    else:
        # We'll plot the top 5 predicted emotions' trajectories
        final_state = traj[-1]
        top_indices = np.argsort(final_state)[-5:][::-1]
        emotion_names = sorted(PLUTCHIK.keys())  # canonical 32-element list

        num_classes = traj.shape[1]
        if num_classes != len(emotion_names):
            st.warning(f"API returned {num_classes} emotion classes, expected {len(emotion_names)}. "
                       "Update the dashboard to match the server's PLUTCHIK constants.")

        fig_traj = go.Figure()
        for idx in top_indices:
            # Bounds check: skip index if either the traj columns or label list is too short
            if idx >= num_classes or idx >= len(emotion_names):
                continue
            fig_traj.add_trace(go.Scatter(
                x=list(range(len(traj))),
                y=traj[:, idx],
                mode='lines',
                name=emotion_names[idx].title(),
                line=dict(width=3)
            ))

        fig_traj.update_layout(
            title="Continuous Emotional State Projection",
            xaxis_title="Continuous Time Steps (dt=0.1)",
            yaxis_title="Probability",
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6edf3'),
            xaxis=dict(showgrid=True, gridcolor='#30363d'),
            yaxis=dict(showgrid=True, gridcolor='#30363d')
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    
    # 3. Sarcasm Signals & Reframing
    col_sig, col_ref = st.columns(2)
    with col_sig:
        st.markdown("#### 🚨 Incongruity Signals")
        if dyn["signals"]:
            for sig in dyn["signals"]:
                st.warning(sig)
        else:
            st.success("No significant incongruity detected.")
            
    with col_ref:
        st.markdown("#### 💡 Strategic Reframe Suggestions")
        if dyn["reframe_suggestions"]:
            for i, sug in enumerate(dyn["reframe_suggestions"]):
                st.info(f"**Option {i+1}:** {sug}")
        else:
            st.write("No reframing required for this input.")
    
    st.divider()

if st.session_state.get("prediction"):
    pred = st.session_state.prediction
    
    # Clean Tabs
    tab_overview, tab_deepdive, tab_internals = st.tabs(["✨ Overview", "📊 Deep Dive", "🔬 Model Internals"])
    
    with tab_overview:
        # Interpretation Summary
        st.markdown("<br>", unsafe_allow_html=True)
        if pred.get("assessment"):
            a = pred["assessment"]
            if a.get("needs_hitl_review"):
                st.warning(
                    f"HITL / review signal: **{', '.join(a.get('reasons', []))}** "
                    f"(confidence band: {a.get('confidence_band', 'n/a')})."
                )
            else:
                st.success("Confidence and sarcasm signals are in a stable band for this prediction.")
        interpretation = f"""
        <div class='glass-card' style='margin-bottom: 2rem;'>
            <h3 style='margin-top: 0; color: var(--accent-primary);'>Inference Insight</h3>
            <p style='font-size: 1.1rem; line-height: 1.6;'>
                The signal indicates <b>{pred['emotion'].title()}</b> with <b>{pred['emotion_confidence']:.1%}</b> statistical confidence. 
                Subtext analysis suggests the intent is <b>{'ironic/hidden' if pred['sarcasm_confidence'] > 0.5 else 'literal'}</b>, 
                operating within the <b>{PLUTCHIK[pred["emotion"]]["ring"].upper()}</b> intensity layer of the Plutchik ecosystem.
            </p>
        </div>
        """
        st.markdown(interpretation, unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: var(--text-secondary); font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;'>Dominant State</p>
                    <h2 style='margin: 0; color: var(--accent-primary); font-size: 2.2rem;'>{pred["emotion"].title()}</h2>
                    <p style='color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>{pred["emotion_confidence"]:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sarcasm_color = "#ff7b72" if pred["sarcasm_confidence"] > 0.5 else "#3fb950"
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: var(--text-secondary); font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;'>Sarcasm Variance</p>
                    <h2 style='margin: 0; color: {sarcasm_color}; font-size: 2.2rem;'>{pred["sarcasm_confidence"]:.1%}</h2>
                    <p style='color: {sarcasm_color}; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>{'DETECTED' if pred["sarcasm_confidence"] > 0.5 else 'LITERAL'}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ring = PLUTCHIK[pred["emotion"]]["ring"]
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: var(--text-secondary); font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;'>Intensity Ring</p>
                    <h2 style='margin: 0; color: var(--text-primary); font-size: 2.2rem;'>{ring.title()}</h2>
                    <p style='color: var(--text-secondary); font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>Plutchik Depth</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: var(--text-secondary); font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;'>Vector Magnitude</p>
                    <h2 style='margin: 0; color: var(--text-primary); font-size: 2.2rem;'>{pred["intensity"]:.2f}</h2>
                    <p style='color: var(--text-secondary); font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>Emotional Force</p>
                </div>
            """, unsafe_allow_html=True)

    with tab_deepdive:
        st.markdown("<br>", unsafe_allow_html=True)
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Plutchik Primary Sectors")
            primary_probs = []
            for emotion in PRIMARY_EMOTIONS:
                emotion_idx = pred["emotion_names"].index(emotion)
                primary_probs.append(pred["emotion_probs"][emotion_idx])
            
            fig_radar = go.Figure(
                data=go.Scatterpolar(
                    r=primary_probs,
                    theta=[e.title() for e in PRIMARY_EMOTIONS],
                    fill='toself',
                    name='Probability',
                    line_color='#58a6ff'
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='#30363d', linecolor='#30363d'),
                    angularaxis=dict(gridcolor='#30363d', linecolor='#30363d')
                ),
                showlegend=False,
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e6edf3')
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with viz_col2:
            st.markdown("#### Intensity Gauge")
            ring = PLUTCHIK[pred["emotion"]]["ring"]
            ring_colors = {
                "mild": "#a371f7",
                "primary": "#58a6ff",
                "intense": "#ff7b72",
                "dyadic": "#3fb950"
            }
            fig_gauge = go.Figure(
                data=go.Indicator(
                    mode="gauge+number",
                    value=pred["intensity"],
                    title={"text": f"Layer: {ring.title()}", "font": {"color": "#e6edf3"}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 1], 'tickcolor': "#e6edf3"},
                        'bar': {'color': ring_colors.get(ring, "#58a6ff")},
                        'bgcolor': "rgba(0,0,0,0)",
                        'bordercolor': "#30363d",
                        'steps': [
                            {'range': [0, 0.25], 'color': "rgba(255,255,255,0.05)"},
                            {'range': [0.25, 0.5], 'color': "rgba(255,255,255,0.1)"},
                            {'range': [0.5, 0.75], 'color': "rgba(255,255,255,0.15)"},
                            {'range': [0.75, 1], 'color': "rgba(255,255,255,0.2)"}
                        ]
                    }
                )
            )
            fig_gauge.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e6edf3'))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown("#### Top 5 Emotions by Confidence")
        top_5_idx = np.argsort(pred["emotion_probs"])[-5:][::-1]
        top_5_emotions = [pred["emotion_names"][i].title() for i in top_5_idx]
        top_5_probs = pred["emotion_probs"][top_5_idx]
        
        fig_bar = go.Figure(
            data=go.Bar(
                y=top_5_emotions,
                x=top_5_probs,
                orientation='h',
                marker_color=['#58a6ff' if e.lower() == pred["emotion"] else '#30363d' for e in top_5_emotions],
                text=[f'{p:.1%}' for p in top_5_probs],
                textposition='auto',
            )
        )
        fig_bar.update_layout(
            xaxis_title="Confidence",
            yaxis_title="",
            height=300,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6edf3'),
            xaxis=dict(showgrid=True, gridcolor='#30363d', zerolinecolor='#30363d'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_internals:
        st.markdown("<br>", unsafe_allow_html=True)
        int_tab1, int_tab2, int_tab3 = st.tabs(["Token Attribution", "Embeddings", "Cosine Similarity"])
        
        with int_tab1:
            if pred["model_type"] == "Nemotron-3 (LLM)":
                st.write("**Nemotron-3 Reasoning Path**")
                st.info(pred.get("reasoning", "No reasoning provided by the model."))
            elif pred.get("attribution_data") and len(pred["attribution_data"]) > 0:
                st.write("**Integrated Gradients: Token Attribution**")
                st.caption("Higher scores indicate words that drove the model toward the predicted emotion. "
                           "Enable “Full explainability” in the sidebar to run Captum on the full [CONTEXT]…[CURRENT]… string.")
                
                attr_df = pd.DataFrame(pred["attribution_data"])
                if "token" in attr_df.columns:
                    attr_df = attr_df[~attr_df["token"].isin(["[PAD]", "<s>", "</s>", "<pad>", "[CONTEXT]", "[/CONTEXT]", "[CURRENT]", "[/CURRENT]"])]
                    
                    fig_attr = px.bar(
                        attr_df, 
                        x="score", 
                        y="token", 
                        orientation='h',
                        color="score",
                        color_continuous_scale="RdBu",
                    )
                    fig_attr.update_layout(
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e6edf3'),
                        xaxis=dict(showgrid=True, gridcolor='#30363d', zerolinecolor='#30363d'),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_attr, use_container_width=True)
                else:
                    st.info("No token attribution data available for this prediction.")

                ctx_top = pred.get("context_span_top") or []
                cur_top = pred.get("current_span_top") or []
                if ctx_top or cur_top:
                    st.markdown("##### Context window (T-2 / T-1) vs current turn — top tokens by |IG|")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.caption("Tokens before [CURRENT] span")
                        st.dataframe(pd.DataFrame(ctx_top), use_container_width=True, height=260)
                    with cc2:
                        st.caption("Tokens inside [CURRENT] span (scenario + topic + utterance)")
                        st.dataframe(pd.DataFrame(cur_top), use_container_width=True, height=260)
                else:
                    st.caption("Context vs current span breakdown appears when running with “Full explainability”.")
            else:
                st.info("Token attribution data is currently unavailable. Enable “Full explainability” in the sidebar (Single Utterance) or use POST /explain.")
        
        with int_tab2:
            st.write("**Embedding Heatmap (Sampled Dims)**")
            emb_blob = pred.get("embedding_info", {}).get("all_token_embeddings")
            if emb_blob is not None and hasattr(emb_blob, "shape") and emb_blob.size > 0:
                heatmap_data = emb_blob
                hm_min = heatmap_data.min(axis=0, keepdims=True)
                hm_max = heatmap_data.max(axis=0, keepdims=True)
                hm_max = np.where(hm_max == hm_min, 1.0, hm_max)
                hm_normalized = (heatmap_data - hm_min) / (hm_max - hm_min + 1e-8)
                
                sample_cols = np.linspace(0, hm_normalized.shape[1] - 1, 30, dtype=int)
                hm_sampled = hm_normalized[:, sample_cols]
                
                fig_hm = go.Figure(
                    data=go.Heatmap(
                        z=hm_sampled,
                        colorscale='Viridis',
                        x=[f'Dim {i}' for i in sample_cols],
                        y=[f'Token {i}' for i in range(hm_sampled.shape[0])]
                    )
                )
                fig_hm.update_layout(
                    height=400, 
                    xaxis_title="Hidden Dims", 
                    yaxis_title="Tokens",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e6edf3')
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.warning("Embedding visualization is only available for the Local RoBERTa model.")
                
        with int_tab3:
            st.write("**Cosine Similarity to Top Emotions**")
            st.caption("Shows how close the prediction is to learned emotion centroids in embedding space.")
            if "embedding_info" in pred and "cls_embedding" in pred["embedding_info"] and emotion_centroids:
                cls_embedding = pred["embedding_info"]["cls_embedding"]
                similarities = {}
                
                # Simple cosine similarity via numpy
                def cosine_sim(a, b):
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

                emotion_names_list = sorted(PLUTCHIK.keys())
                for emotion_idx, centroid in emotion_centroids.items():
                    sim = cosine_sim(cls_embedding, centroid)
                    emotion_str = emotion_names_list[emotion_idx] if isinstance(emotion_idx, (int, np.integer)) else emotion_idx
                    similarities[emotion_str] = sim
                
                sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
                sim_emotions, sim_values = zip(*sorted_sims)
                
                fig_sim = go.Figure(
                    data=go.Bar(
                        x=list(sim_values),
                        y=[e.title() for e in sim_emotions],
                        orientation='h',
                        marker_color=['#58a6ff' if e == pred["emotion"] else '#30363d' for e in sim_emotions],
                        text=[f'{v:.3f}' for v in sim_values],
                        textposition='auto'
                    )
                )
                fig_sim.update_layout(
                    xaxis_title="Cosine Similarity",
                    yaxis_title="",
                    height=350,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e6edf3'),
                    xaxis=dict(showgrid=True, gridcolor='#30363d', zerolinecolor='#30363d'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.warning("Cosine similarity analysis requires the Local RoBERTa model's learned centroids.")


# ============== SIDEBAR: HISTORY & ABOUT ==============
st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Prediction History")

if st.session_state.prediction:
    # Add to history if not already the latest
    latest = st.session_state.prediction
    if not st.session_state.history or st.session_state.history[0]["text"] != latest.get("text", "Unknown text"):
        st.session_state.history.insert(0, {
            "emotion": latest["emotion"],
            "confidence": latest["emotion_confidence"],
            "text": latest.get("text", "Unknown text")[:30] + "..."
        })

for h in st.session_state.history[:5]:
    st.sidebar.markdown(f"""
        <div style='background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid var(--accent-primary);'>
            <p style='margin:0; font-size: 0.85rem; color: var(--text-secondary);'>{html.escape(h['text'])}</p>
            <p style='margin:0; font-weight: 600; color: var(--text-primary); margin-top: 4px;'>{html.escape(h['emotion'].title())} ({h['confidence']:.0%})</p>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 The Plutchik Lexicon")
with st.sidebar.expander("Explore Emotional Layers"):
    st.markdown("""
        <div style='font-size: 0.85rem;'>
            <p><b style='color: #ff7b72;'>Intense:</b> Raw, visceral reactions (Rage, Grief, Terror).</p>
            <p><b style='color: #58a6ff;'>Primary:</b> Balanced, conscious states (Anger, Sadness, Fear).</p>
            <p><b style='color: #a371f7;'>Mild:</b> Subtle, transient feelings (Annoyance, Pensiveness, Apprehension).</p>
            <p><b style='color: #3fb950;'>Dyadic:</b> Complex blends (Contempt, Remorse, Love).</p>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.info("""
The **Plutchik Wheel** defines emotions as a spectrum of 32 classes. 
This AI decodes the **subtext**—detecting when words and intent diverge.
""")

st.sidebar.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.7rem; margin-top: 2rem;'>
    <p style='margin-bottom: 0;'>Version 2.1 Production Edition</p>
    <p style='margin-top: 0;'>© 2026 Plutchik ERC Project</p>
</div>
""", unsafe_allow_html=True)
