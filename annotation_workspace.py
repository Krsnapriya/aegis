"""
Annotation Workspace for Plutchik ERC
Collaborative HITL annotation tool with real-time IAA calculation
"""
import streamlit as st
import requests
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import hashlib

# All 32 emotion classes
EMOTION_CLASSES = [
    'ecstasy', 'admiration', 'terror', 'amazement', 'grief', 'loathing',
    'rage', 'vigilance', 'joy', 'trust', 'fear', 'surprise', 'sadness',
    'disgust', 'anger', 'anticipation', 'serenity', 'acceptance',
    'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance',
    'interest', 'optimism', 'love', 'submission', 'awe', 'disapproval',
    'remorse', 'contempt', 'aggressiveness'
]

INTENSITY_LABELS = ['mild', 'primary', 'intense']

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Plutchik Annotation Workspace", layout="wide")

st.title("🏷️ Plutchik ERC Annotation Workspace")
st.markdown("**Human-in-the-Loop annotation tool** with real-time IAA tracking")

# Sidebar
st.sidebar.header("Settings")
annotator_id = st.sidebar.text_input("Annotator ID", value=f"annotator_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]}")
api_key = st.sidebar.text_input("API Key", value="demo-key-123")

# Load existing annotations
ANNOTATIONS_FILE = Path("/workspace/data/annotations.jsonl")
CORRECTIONS_FILE = Path("/workspace/data/corrections.jsonl")

def load_annotations():
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return [json.loads(line) for line in f]
    return []

def save_annotation(annotation):
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, 'a') as f:
        f.write(json.dumps(annotation) + '\n')

def calculate_iaa(annotations):
    """Calculate Inter-Annotator Agreement for samples with multiple annotations"""
    # Group by text hash
    by_text = {}
    for ann in annotations:
        text_hash = hashlib.md5(ann['text'].encode()).hexdigest()
        if text_hash not in by_text:
            by_text[text_hash] = []
        by_text[text_hash].append(ann)
    
    # Calculate agreement rates
    multi_annotated = {k: v for k, v in by_text.items() if len(v) > 1}
    
    if not multi_annotated:
        return {'samples_multi_annotated': 0, 'emotion_agreement': None, 'sarcasm_agreement': None}
    
    emotion_agreements = []
    sarcasm_agreements = []
    
    for text_hash, anns in multi_annotated.items():
        emotions = [a['emotion'] for a in anns]
        sarcsms = [a['sarcasm'] for a in anns]
        
        # Pairwise agreement
        for i in range(len(emotions)):
            for j in range(i+1, len(emotions)):
                emotion_agreements.append(1 if emotions[i] == emotions[j] else 0)
                sarcasm_agreements.append(1 if sarcsms[i] == sarcsms[j] else 0)
    
    return {
        'samples_multi_annotated': len(multi_annotated),
        'emotion_agreement': sum(emotion_agreements) / len(emotion_agreements) if emotion_agreements else None,
        'sarcasm_agreement': sum(sarcasm_agreements) / len(sarcasm_agreements) if sarcasm_agreements else None,
        'total_annotations': len(annotations)
    }

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Annotate", "📊 IAA Dashboard", "🔍 Review", "📁 Export"])

with tab1:
    st.header("New Annotation")
    
    # Option 1: Enter text manually
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio("Input method:", ["Manual entry", "From corrections queue", "From training data"])
        
        text_to_annotate = ""
        model_prediction = None
        
        if input_method == "Manual entry":
            text_to_annotate = st.text_area("Enter text to annotate:", height=100)
        
        elif input_method == "From corrections queue":
            if CORRECTIONS_FILE.exists():
                with open(CORRECTIONS_FILE) as f:
                    corrections = [json.loads(line) for line in f]
                if corrections:
                    selected_idx = st.selectbox("Select correction sample:", range(len(corrections)), 
                                               format_func=lambda i: f"[{i}] {corrections[i]['text'][:60]}...")
                    text_to_annotate = corrections[selected_idx]['text']
                    model_prediction = corrections[selected_idx].get('model_prediction')
                    st.info(f"Model predicted: {model_prediction}")
                    st.write(f"Human label: {corrections[selected_idx]['true_emotion']}")
            else:
                st.warning("No corrections file found")
        
        elif input_method == "From training data":
            with open("/workspace/data/train.jsonl") as f:
                train_data = [json.loads(line) for line in f]
            selected_idx = st.selectbox("Select training sample:", range(len(train_data)),
                                       format_func=lambda i: f"[{i}] {train_data[i]['text'][:60]}...")
            text_to_annotate = train_data[selected_idx]['text']
            model_prediction = train_data[selected_idx].get('emotion')
            st.info(f"Existing label: {train_data[selected_idx]['emotion']} (IAA: {train_data[selected_idx].get('iaa_score', 'N/A')})")
    
    with col2:
        if text_to_annotate and input_method == "Manual entry":
            st.subheader("Model Prediction")
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_to_annotate},
                    headers={"X-API-Key": api_key}
                )
                model_prediction = response.json()
                
                # Show top 3 emotions
                sorted_emotions = sorted(model_prediction['all_emotions'].items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                for emo, conf in sorted_emotions:
                    st.progress(conf)
                    st.caption(f"{emo}: {conf:.1%}")
                
                st.metric("Sarcasm", f"{'⚠️' if model_prediction['sarcasm'] else '✓'}", 
                         f"{model_prediction['sarcasm_score']:.1%}")
                st.metric("Intensity", model_prediction['intensity'])
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # Annotation form
    if text_to_annotate:
        st.divider()
        st.subheader("Your Annotation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_emotion = st.selectbox("Primary Emotion (32-class):", EMOTION_CLASSES)
        
        with col2:
            selected_intensity = st.selectbox("Intensity Ring:", INTENSITY_LABELS)
        
        with col3:
            selected_sarcasm = st.checkbox("Sarcastic?")
        
        confidence_rating = st.slider("Your confidence in this annotation:", 0.0, 1.0, 0.7, 0.05)
        
        notes = st.text_area("Additional notes (optional):", placeholder="Why did you choose this label?")
        
        if st.button("Submit Annotation", type="primary"):
            annotation = {
                'text': text_to_annotate,
                'emotion': selected_emotion,
                'intensity': selected_intensity,
                'sarcasm': selected_sarcasm,
                'annotator_id': annotator_id,
                'confidence': confidence_rating,
                'notes': notes,
                'timestamp': datetime.now().isoformat(),
                'model_prediction': model_prediction
            }
            
            save_annotation(annotation)
            st.success("✅ Annotation saved!")
            
            # Show IAA update
            all_anns = load_annotations()
            iaa_stats = calculate_iaa(all_anns)
            if iaa_stats['emotion_agreement'] is not None:
                st.info(f"Current IAA - Emotion: {iaa_stats['emotion_agreement']:.1%}, Sarcasm: {iaa_stats['sarcasm_agreement']:.1%}")

with tab2:
    st.header("Inter-Annotator Agreement Dashboard")
    
    all_annotations = load_annotations()
    iaa_stats = calculate_iaa(all_annotations)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Annotations", len(all_annotations))
    
    with col2:
        st.metric("Multi-Annotated Samples", iaa_stats.get('samples_multi_annotated', 0))
    
    with col3:
        emotion_iaa = iaa_stats.get('emotion_agreement')
        if emotion_iaa:
            st.metric("Emotion IAA", f"{emotion_iaa:.1%}", 
                     delta="Good" if emotion_iaa > 0.7 else "Needs Review" if emotion_iaa < 0.5 else "OK")
        else:
            st.metric("Emotion IAA", "N/A")
    
    with col4:
        sarcasm_iaa = iaa_stats.get('sarcasm_agreement')
        if sarcasm_iaa:
            st.metric("Sarcasm IAA", f"{sarcasm_iaa:.1%}",
                     delta="Good" if sarcasm_iaa > 0.8 else "Needs Review" if sarcasm_iaa < 0.6 else "OK")
        else:
            st.metric("Sarcasm IAA", "N/A")
    
    # Per-emotion breakdown
    if all_annotations:
        st.divider()
        st.subheader("Per-Emotion Statistics")
        
        df = pd.DataFrame(all_annotations)
        emotion_counts = df['emotion'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(emotion_counts)
        
        with col2:
            # Show ambiguous cases (low agreement)
            st.write("**Samples needing review (disagreements):**")
            by_text = {}
            for ann in all_annotations:
                text_hash = hashlib.md5(ann['text'].encode()).hexdigest()
                if text_hash not in by_text:
                    by_text[text_hash] = []
                by_text[text_hash].append(ann)
            
            disagreements = []
            for text_hash, anns in by_text.items():
                if len(anns) > 1:
                    emotions = set(a['emotion'] for a in anns)
                    if len(emotions) > 1:
                        disagreements.append({
                            'text': anns[0]['text'][:80] + '...',
                            'emotions': list(emotions),
                            'annotators': len(anns)
                        })
            
            if disagreements:
                for d in disagreements[:10]:
                    st.warning(f"\"{d['text']}\" - Disagreement: {', '.join(d['emotions'])}")
            else:
                st.success("No disagreements found!")

with tab3:
    st.header("Review Annotations")
    
    all_annotations = load_annotations()
    
    if all_annotations:
        # Filter options
        filter_emotion = st.multiselect("Filter by emotion:", EMOTION_CLASSES)
        filter_annotator = st.text_input("Filter by annotator ID:")
        filter_sarcasm = st.selectbox("Sarcasm filter:", ["All", "Sarcastic only", "Non-sarcastic only"])
        
        filtered = all_annotations
        
        if filter_emotion:
            filtered = [a for a in filtered if a['emotion'] in filter_emotion]
        
        if filter_annotator:
            filtered = [a for a in filtered if filter_annotator in a['annotator_id']]
        
        if filter_sarcasm == "Sarcastic only":
            filtered = [a for a in filtered if a.get('sarcasm')]
        elif filter_sarcasm == "Non-sarcastic only":
            filtered = [a for a in filtered if not a.get('sarcasm')]
        
        st.write(f"Showing {len(filtered)} of {len(all_annotations)} annotations")
        
        # Display as table
        if filtered:
            display_data = []
            for a in filtered[-50:]:  # Last 50
                display_data.append({
                    'Text': a['text'][:60] + '...' if len(a['text']) > 60 else a['text'],
                    'Emotion': a['emotion'],
                    'Intensity': a['intensity'],
                    'Sarcasm': '⚠️' if a.get('sarcasm') else '✓',
                    'Annotator': a['annotator_id'][-6:],
                    'Confidence': f"{a.get('confidence', 0):.0%}"
                })
            
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
            
            # Click to view full details
            selected = st.selectbox("View full details:", range(len(filtered)),
                                   format_func=lambda i: f"[{len(filtered)-1-i}] {filtered[len(filtered)-1-i]['text'][:50]}...")
            
            if selected is not None:
                ann = filtered[len(filtered)-1-selected]
                st.expander("Full annotation details", expanded=True).write(ann)
    else:
        st.info("No annotations yet. Start annotating in the first tab!")

with tab4:
    st.header("Export Data")
    
    all_annotations = load_annotations()
    
    if all_annotations:
        st.write(f"**{len(all_annotations)}** annotations ready for export")
        
        # Export formats
        export_format = st.selectbox("Export format:", ["JSONL", "CSV", "JSON"])
        
        if st.button("Download Export"):
            if export_format == "JSONL":
                jsonl_content = '\n'.join(json.dumps(a) for a in all_annotations)
                st.download_button(
                    label="Download JSONL",
                    data=jsonl_content,
                    file_name=f"plutchik_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                    mime="text/plain"
                )
            
            elif export_format == "CSV":
                df = pd.DataFrame(all_annotations)
                csv_content = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name=f"plutchik_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "JSON":
                json_content = json.dumps(all_annotations, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_content,
                    file_name=f"plutchik_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Stats for retraining
        st.divider()
        st.subheader("Dataset Statistics for Retraining")
        
        emotion_dist = pd.Series([a['emotion'] for a in all_annotations]).value_counts()
        st.write("**Emotion distribution:**")
        st.bar_chart(emotion_dist)
        
        intensity_dist = pd.Series([a['intensity'] for a in all_annotations]).value_counts()
        st.write("**Intensity distribution:**")
        st.bar_chart(intensity_dist)
        
        sarcasm_rate = sum(1 for a in all_annotations if a.get('sarcasm')) / len(all_annotations) * 100
        st.metric("Sarcasm rate", f"{sarcasm_rate:.1f}%")
    else:
        st.info("No annotations to export yet.")

# Footer
st.divider()
st.caption(f"Annotations file: {ANNOTATIONS_FILE.absolute()} | Correlations file: {CORRECTIONS_FILE.absolute() if CORRECTIONS_FILE.exists() else 'Not created yet'}")
