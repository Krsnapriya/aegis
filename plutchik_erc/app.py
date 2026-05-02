import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Plutchik ERC Dashboard", layout="wide")

st.title("🎨 Plutchik Emotion Recognition Dashboard")
st.markdown("**32-class emotion detection** with sarcasm, intensity, and dialogue analysis")

# Sidebar
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("API Key", value="demo-key-123")
session_id = st.sidebar.text_input("Session ID", value="default")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Prediction", "💬 Conversation Arc", "📁 Batch Analysis", "ℹ️ About"])

with tab1:
    st.header("Single Utterance Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Type or paste your text here..."
        )
        
        if st.button("Analyze", type="primary"):
            if text_input:
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"text": text_input, "session_id": session_id},
                            headers={"X-API-Key": api_key}
                        )
                        result = response.json()
                        st.session_state['last_result'] = result
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text")
    
    with col2:
        if 'last_result' in st.session_state:
            r = st.session_state['last_result']
            
            # Emotion radar chart
            emotions = list(r['all_emotions'].keys())
            values = list(r['all_emotions'].values())
            
            # Order for Plutchik wheel (approximate)
            order = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 
                     'anger', 'anticipation', 'ecstasy', 'admiration', 'terror', 
                     'amazement', 'grief', 'loathing', 'rage', 'vigilance',
                     'serenity', 'acceptance', 'apprehension', 'distraction',
                     'pensiveness', 'boredom', 'annoyance', 'interest',
                     'optimism', 'love', 'submission', 'awe', 'disapproval',
                     'remorse', 'contempt', 'aggressiveness']
            
            # Reorder values
            ordered_values = [values[emotions.index(e)] if e in emotions else 0 for e in order]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=ordered_values,
                theta=order,
                fill='toself',
                line_color='#636EFA'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(ordered_values)*1.2])),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            st.metric("Primary Emotion", f"{r['emotion']} ({r['confidence']:.1%})")
            st.metric("Sarcasm", f"{'⚠️ Yes' if r['sarcasm'] else '✓ No'} ({r['sarcasm_score']:.1%})")
            st.metric("Intensity Ring", r['intensity'])
    
    # Detailed results
    if 'last_result' in st.session_state:
        r = st.session_state['last_result']
        
        st.subheader("Detailed Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Top 5 Emotions:**")
            sorted_emotions = sorted(r['all_emotions'].items(), key=lambda x: x[1], reverse=True)[:5]
            for emo, conf in sorted_emotions:
                st.progress(conf)
                st.caption(f"{emo}: {conf:.1%}")
        
        with col2:
            st.write("**Intensity Distribution:**")
            for intensity, score in r['intensity_scores'].items():
                st.progress(score)
                st.caption(f"{intensity}: {score:.1%}")
        
        with col3:
            st.write("**Analysis:**")
            st.info(f"""
            - **Detected**: {r['emotion']}
            - **Confidence**: {r['confidence']:.1%}
            - **Sarcasm Score**: {r['sarcasm_score']:.1%}
            - **Intensity**: {r['intensity']}
            """)

with tab2:
    st.header("Conversation Emotion Arc")
    st.markdown("Analyze emotional trajectory across multiple turns")
    
    # Sample conversation
    default_conv = """I'm really excited about this project.
Yes, me too! The possibilities are endless.
But I'm a bit worried about the deadline.
Don't worry, we've got this.
Actually, I'm starting to panic now.
Hey, deep breaths. We'll figure it out together."""
    
    conv_text = st.text_area("Enter conversation (one utterance per line):", 
                             value=default_conv, height=200)
    
    if st.button("Analyze Conversation Arc"):
        turns = [line.strip() for line in conv_text.split('\n') if line.strip()]
        
        if turns:
            with st.spinner("Analyzing conversation arc..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/arc",
                        json=turns,
                        headers={"X-API-Key": api_key}
                    )
                    arc_result = response.json()
                    st.session_state['arc_result'] = arc_result
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if 'arc_result' in st.session_state:
        arc = st.session_state['arc_result']
        
        # Emotion timeline
        turns = [t['turn'] for t in arc['trajectory']]
        emotions_seq = [t['emotion'] for t in arc['trajectory']]
        confidences = [t['confidence'] for t in arc['trajectory']]
        
        # Color mapping for emotions
        emotion_colors = {
            'joy': '#FFD700', 'trust': '#90EE90', 'fear': '#8B0000',
            'surprise': '#FFA500', 'sadness': '#00008B', 'disgust': '#006400',
            'anger': '#DC143C', 'anticipation': '#9370DB', 'ecstasy': '#FF69B4',
            'admiration': '#DA70D6', 'terror': '#8B0000', 'amazement': '#FF4500',
            'grief': '#2F4F4F', 'loathing': '#8B4513', 'rage': '#B22222',
            'vigilance': '#556B2F'
        }
        
        colors = [emotion_colors.get(e, '#808080') for e in emotions_seq]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=turns, y=confidences,
            mode='markers+lines',
            marker=dict(size=15, color=colors),
            text=[f"Turn {t}: {e}<br>Conf: {c:.1%}" for t, e, c in zip(turns, emotions_seq, confidences)],
            hoverinfo='text'
        ))
        
        # Mark inflection points
        inflections = [t for t in arc['trajectory'] if t.get('inflection_point')]
        if inflections:
            inflect_turns = [t['turn'] for t in inflections]
            inflect_conf = [t['confidence'] for t in inflections]
            fig.add_trace(go.Scatter(
                x=inflect_turns, y=inflect_conf,
                mode='markers',
                marker=dict(size=20, color='red', symbol='x'),
                name='Inflection Point'
            ))
        
        fig.update_layout(
            title="Emotional Trajectory Across Conversation",
            xaxis_title="Turn Number",
            yaxis_title="Confidence",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Turn-by-turn breakdown
        st.subheader("Turn-by-Turn Analysis")
        for turn_data in arc['trajectory']:
            with st.expander(f"Turn {turn_data['turn']}: {turn_data['emotion']} ({turn_data['confidence']:.1%})"):
                st.write(f"**Intensity**: {turn_data['intensity']}")
                st.write(f"**Sarcasm**: {'⚠️ Yes' if turn_data['sarcasm'] else '✓ No'} ({turn_data['sarcasm_score']:.1%})")
                if turn_data.get('inflection_point'):
                    st.warning("⚠️ **Emotional Inflection Point** - Significant shift detected!")

with tab3:
    st.header("Batch Analysis")
    st.markdown("Upload a CSV file with a 'text' column for batch processing")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        
        if 'text' in df.columns:
            st.write(f"Found {len(df)} texts to analyze")
            
            if st.button("Run Batch Analysis"):
                texts = df['text'].tolist()[:50]  # Limit to 50 for demo
                
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    results = []
                    for text in texts:
                        try:
                            resp = requests.post(
                                f"{API_URL}/predict",
                                json={"text": str(text)},
                                headers={"X-API-Key": api_key}
                            )
                            results.append(resp.json())
                        except:
                            results.append(None)
                    
                    # Add results to dataframe
                    df_batch = df.head(len(texts)).copy()
                    df_batch['emotion'] = [r['emotion'] if r else None for r in results]
                    df_batch['confidence'] = [r['confidence'] if r else None for r in results]
                    df_batch['sarcasm'] = [r['sarcasm'] if r else None for r in results]
                    
                    st.success("Analysis complete!")
                    st.dataframe(df_batch)
                    
                    # Show distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Emotion Distribution:**")
                        emotion_counts = df_batch['emotion'].value_counts()
                        st.bar_chart(emotion_counts)
                    with col2:
                        st.write("**Sarcasm Rate:**")
                        sarcasm_rate = df_batch['sarcasm'].sum() / len(df_batch) * 100
                        st.metric("Sarcasm Rate", f"{sarcasm_rate:.1f}%")
        else:
            st.error("CSV must contain a 'text' column")

with tab4:
    st.header("About Plutchik ERC")
    
    st.markdown("""
    ### 32-Class Emotion Model
    
    Based on Robert Plutchik's Wheel of Emotions, this system detects:
    
    **8 Primary Emotions:**
    - Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
    
    **24 Secondary Emotions** (combinations and intensity variations):
    - Ecstasy, Admiration, Terror, Amazement, Grief, Loathing, Rage, Vigilance
    - Serenity, Acceptance, Apprehension, Distraction, Pensiveness, Boredom
    - Annoyance, Interest, Optimism, Love, Submission, Awe, Disapproval, Remorse, Contempt, Aggressiveness
    
    ### Additional Features
    
    - **Sarcasm Detection**: Binary classification with confidence scores
    - **Intensity Rings**: Mild → Primary → Intense
    - **Dialogue Context**: Sliding window of previous turns
    - **Emotion Arc Tracking**: KL divergence-based inflection point detection
    
    ### API Endpoints
    
    - `POST /predict` - Single utterance analysis
    - `POST /predict/batch` - Batch processing
    - `POST /predict/arc` - Conversation trajectory
    - `POST /explain` - Token-level attribution
    - `POST /correct` - Submit HITL corrections
    - `GET /health` - System status
    
    ### Privacy
    
    All inference runs locally. No data leaves your browser/device.
    """)
    
    st.info("Built for the Hackathon - 18 hour sprint")
