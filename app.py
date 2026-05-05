import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import json
import torch
import time
from datetime import datetime
from dotenv import load_dotenv

# ============== ARCHITECTURAL UNIFICATION ==============
# This app is optimized for Hugging Face Spaces. 
# It unifies the FastAPI inference engine and Streamlit dashboard into a single process.
# ========================================================

# Load environment variables
project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

# Standard imports from the repo
from utils.constants import PLUTCHIK, EMOTION_NAMES, NUM_EMOTIONS, RING_INTENSITY
from utils.preprocessing import ERCPreprocessor
from utils.llm_inference import NemotronClient
from core.advanced_engine import AdvancedPlutchikEngine, InputSanitizer
from models.multitask_emotion_model import PluTchikMultiTaskModel
from utils.explainability_v2 import CaptumExplainer

# ============== HF SPACE CONFIG ==============
st.set_page_config(
    page_title="Plutchik ERC Engine",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    :root {
        --accent-primary: #58a6ff;
        --accent-secondary: #1f6feb;
        --bg-dark: #0d1117;
        --card-bg: #161b22;
        --text-main: #e6edf3;
    }
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-main);
    }
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# ============== CORE ENGINE LOADER (CACHED) ==============
@st.cache_resource
def load_plutchik_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Initializing Plutchik Engine on {device} ---")
    
    # 1. Initialize Model
    model = PluTchikMultiTaskModel(num_emotions=NUM_EMOTIONS)
    model_path = project_dir / "my_plutchik_model" / "best_model.pt"
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"✓ Weights loaded from {model_path}")
    else:
        print("⚠ No weights found. Running in base-mode.")
        
    model.to(device).eval()
    
    # 2. Preprocessor & Explainer
    preprocessor = ERCPreprocessor(PLUTCHIK)
    explainer = CaptumExplainer(model, preprocessor.tokenizer)
    
    # 3. Engines
    engine = AdvancedPlutchikEngine(base_model=model, tokenizer=preprocessor.tokenizer, device=device)
    sanitizer = InputSanitizer()
    
    return engine, sanitizer, explainer, preprocessor

# Load the engine components
engine, sanitizer, explainer, preprocessor = load_plutchik_engine()
llm_client = NemotronClient()

# ============== SESSION STATE INITIALIZATION ==============
if "history_buffer" not in st.session_state:
    st.session_state.history_buffer = []
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ============== SIDEBAR COMMAND CENTER ==============
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Plutchik-wheel.svg/1024px-Plutchik-wheel.svg.png", width=150)
    st.title("🎭 Control Console")
    st.markdown("---")
    
    model_core = st.radio(
        "🧠 Analysis Core",
        ["Local RoBERTa", "Nemotron-3 (LLM)", "Compare Both Models"],
        help="Select the inference backbone. Nemotron-3 provides grounded reasoning."
    )
    
    analysis_mode = st.selectbox(
        "🧭 Protocol",
        ["Single Utterance", "Conversation Arc", "Dynamic Intelligence"],
        index=0
    )
    
    st.markdown("---")
    with st.expander("🛠️ Context Matrix"):
        scenario = st.text_input("Scenario", "general")
        topic = st.text_input("Topic", "general")
        speaker = st.text_input("Speaker", "user")
        use_history = st.checkbox("Include History Buffer", value=True)

# ============== MAIN APP INTERFACE ==============
st.title("Plutchik Emotion Engine")
st.caption(f"v2.1 Production Edition | Powered by {model_core}")

# Input Section
user_text = st.text_area("📡 Input Signal", placeholder="Type or paste text to analyze emotional subtext...", height=100)

col_actions, col_meta = st.columns([1, 1])

with col_actions:
    if st.button("✨ Run Analysis", type="primary", use_container_width=True):
        if not user_text:
            st.warning("Please input signal first.")
        else:
            with st.spinner("Processing emotional DNA..."):
                # 1. Sanitization
                is_valid, sanitized_text, reason, emoji_emotion = sanitizer.sanitize_and_validate(user_text)
                
                if not is_valid:
                    st.error(f"❌ Input Rejected: {reason}")
                else:
                    # 2. Context Preparation
                    context = " | ".join(st.session_state.history_buffer[-3:]) if use_history and st.session_state.history_buffer else "[NO_CONTEXT]"
                    
                    # 3. Execution Logic (Direct Method Calls)
                    if model_core == "Local RoBERTa":
                        # Direct Engine Inference
                        result = engine.predict(sanitized_text, scenario, topic, context)
                        
                        # Add explanations if single utterance
                        if analysis_mode == "Single Utterance":
                            expl_data = explainer.explain_prediction(sanitized_text, context)
                            result["explanations"] = expl_data
                        
                        st.session_state.prediction = result
                        st.session_state.prediction["model_type"] = "Local RoBERTa"
                        
                    elif model_core == "Nemotron-3 (LLM)":
                        llm_res = llm_client.predict_emotion(sanitized_text, scenario, topic, context)
                        if "error" in llm_res:
                            st.error(f"❌ LLM Error: {llm_res['error']}")
                        else:
                            st.session_state.prediction = {
                                "emotion": llm_res["emotion"],
                                "confidence": llm_res.get("confidence", 1.0),
                                "sarcasm_prob": llm_res.get("sarcasm_confidence", 0.0),
                                "intensity": llm_res.get("intensity", 0.5),
                                "reasoning": llm_res.get("reasoning"),
                                "model_type": "Nemotron-3 (LLM)",
                                "emotion_probs": [1.0 if EMOTION_NAMES[i] == llm_res["emotion"] else 0.0 for i in range(NUM_EMOTIONS)]
                            }
                            
                    elif model_core == "Compare Both Models":
                        local_res = engine.predict(sanitized_text, scenario, topic, context)
                        llm_res = llm_client.predict_emotion(sanitized_text, scenario, topic, context)
                        
                        st.session_state.comparison = {
                            "local": local_res,
                            "llm": llm_res
                        }
                        st.session_state.prediction = local_res
                        
                    # Update History
                    st.session_state.history_buffer.append(f"{speaker}: {sanitized_text}")
                    st.success("✓ Analysis Complete")

# ============== RESULTS DISPLAY ==============
if st.session_state.prediction:
    pred = st.session_state.prediction
    
    st.markdown("---")
    t1, t2 = st.tabs(["📊 Summary", "🔬 Deep Dive"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant Emotion", pred["emotion"].title())
        c2.metric("Intensity", f"{pred['intensity']:.2f}")
        c3.metric("Sarcasm Prob", f"{pred['sarcasm_prob']:.1%}")
        
        # Interpretation
        st.markdown(f"**Interpretation:** {pred.get('reasoning', 'Analysis complete.')}")
        
        # Plutchik Vector Chart
        fig = px.bar(
            x=EMOTION_NAMES, 
            y=pred.get("emotion_probs", [0]*32), 
            title="Emotional Probability Distribution",
            labels={'x': 'Emotion', 'y': 'Probability'},
            color_discrete_sequence=['#58a6ff']
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e6edf3'))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        if "explanations" in pred:
            st.markdown("#### Token Attribution (Captum)")
            st.write("Heatmap of word importance for the predicted emotion:")
            st.json(pred["explanations"].get("token_attributions", []))
        else:
            st.info("Detailed explainability heatmaps are optimized for Local RoBERTa mode.")

# Footer
st.markdown("---")
st.caption("HF Space Build | Unified Plutchik ERC Architecture")
