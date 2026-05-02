# Plutchik ERC - Implementation Status & Roadmap

## Executive Summary

**Current State:** Functional MVP with single-utterance prediction, basic conversation arc analysis, and batch processing.

**Missing Critical Features:** Dialogue-level emotion arcs, comparative analysis, proper Captum explainability, browser plugin, speaker profiling, HITL annotation workspace.

**Technical Debt:** Model path hardcoded to `/workspace/plutchik_erc/my_plutchik_model/best_model.pt` but actual path is `/workspace/my_plutchik_model/best_model.pt`.

---

## Surface 1: Analyst Dashboard

### ✅ Implemented
- [x] Text input → 32-class prediction
- [x] Radar chart visualization (Plotly)
- [x] Sarcasm bar indicator
- [x] Intensity gauge (mild/primary/intense)
- [x] Basic token attribution (simplified, not Captum)
- [x] Conversation arc timeline (turn number vs emotion)
- [x] Batch CSV upload with emotion distribution

### ❌ Missing
- [ ] **Dialogue-level emotion arc**: Current `/predict/arc` shows turn-by-turn but doesn't detect inflection points meaningfully (KL threshold is arbitrary 0.5)
- [ ] **Comparative analysis**: No side-by-side conversation comparison view
- [ ] **Captum integration**: Current `/explain` uses crude word-length heuristic, not Integrated Gradients
- [ ] **Context window visualization**: No panel showing how previous 2 turns influenced prediction
- [ ] **Statistical emotion profile**: Batch mode doesn't show sarcasm rate by speaker role or average IAA by scenario
- [ ] **Emotion ring coloring**: Timeline uses hardcoded colors for only 16 emotions, not full 32

### Priority Fixes Needed

#### 1. Fix Model Path
```python
# inference_server.py line 49
checkpoint = torch.load('/workspace/my_plutchik_model/best_model.pt', ...)
```

#### 2. Add Comparative Analysis Endpoint
```python
@app.post("/compare")
async def compare_conversations(
    conversation_a: List[str],
    conversation_b: List[str]
):
    """Side-by-side emotion trajectory comparison"""
    arc_a = await predict_arc(conversation_a)
    arc_b = await predict_arc(conversation_b)
    
    # Calculate divergence metrics
    divergence_points = []
    for turn_a, turn_b in zip(arc_a['trajectory'], arc_b['trajectory']):
        kl_div = compute_kl_divergence(turn_a['all_emotions'], turn_b['all_emotions'])
        if kl_div > 0.3:
            divergence_points.append({
                'turn': turn_a['turn'],
                'divergence': kl_div,
                'emotion_a': turn_a['emotion'],
                'emotion_b': turn_b['emotion']
            })
    
    return {
        'conversation_a': arc_a,
        'conversation_b': arc_b,
        'divergence_points': divergence_points,
        'overall_similarity': 1.0 - np.mean([d['divergence'] for d in divergence_points])
    }
```

#### 3. Integrate Captum for Real Explainability
```python
from captum.attr import IntegratedGradients

def get_token_attributions(text: str, target_class: int):
    model.eval()
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    input_ids.requires_grad_(True)
    
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        input_ids,
        target=target_class,
        additional_forward_args=(inputs['attention_mask'].to(device),),
        return_convergence_delta=True
    )
    
    # Map back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return [{'token': t, 'attribution': a.item()} 
            for t, a in zip(tokens, attributions[0])]
```

---

## Surface 2: Browser Plugin (North Star)

### ✅ Implemented
- [x] Architecture documented (Shadow DOM, WebGPU/WASM)
- [x] Privacy-first design (on-device inference)

### ❌ Missing
- [ ] **Reddit integration**: No Chrome extension code exists
- [ ] **Emotion overlay dots**: No injection logic for comment threads
- [ ] **Sarcasm flag UI**: No visual indicator for sarcastic comments
- [ ] **Gmail integration**: No email emotion detection
- [ ] **LinkedIn integration**: No professional tone advisor
- [ ] **ONNX export**: No script to convert PyTorch model to ONNX for Transformers.js
- [ ] **Transformers.js integration**: No browser-compatible model

### Required Files to Create
```
extension/
├── manifest.json          # Chrome extension config
├── background.js          # Service worker
├── content/
│   ├── reddit.js         # Reddit DOM injection
│   ├── gmail.js          # Gmail integration
│   └── linkedin.js       # LinkedIn tone advisor
├── popup/
│   ├── popup.html
│   ├── popup.js
│   └── styles.css
├── models/
│   └── plutchik.onnx     # Exported model (needs creation)
└── utils/
    ├── inference.js      # Transformers.js wrapper
    └── emotion-colors.js # Ring color mapping
```

### Model Export Script Needed
```python
# scripts/export_onnx.py
import torch
from models.multitask_emotion_model import PluTchikMultiTaskModel

model = PluTchikMultiTaskModel()
checkpoint = torch.load('my_plutchik_model/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy input for tracing
dummy_input = torch.ones(1, 128, dtype=torch.long)
dummy_attention = torch.ones(1, 128, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_input, dummy_attention),
    "extension/models/plutchik.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['emotion_logits', 'sarcasm_logits', 'intensity_logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'}
    }
)
```

---

## Surface 3: Inference API (B2B Platform Layer)

### ✅ Implemented
- [x] `/predict` - Full multi-task output
- [x] `/health` - System status
- [x] `/predict/batch` - Array of utterances
- [x] `/predict/arc` - Emotion trajectory with inflection detection
- [x] `/explain` - Token attribution (simplified)
- [x] `/correct` - HITL correction logging
- [x] Session management with sliding context window
- [x] Rate limiting with API keys

### ❌ Missing
- [ ] **Proper Captum endpoint**: Current `/explain` is placeholder
- [ ] **API key management**: Only one hardcoded demo key
- [ ] **Rate limit persistence**: In-memory deque resets on restart
- [ ] **Usage analytics**: No tracking of API calls per key
- [ ] **Authentication**: No JWT or OAuth, just header-based API key
- [ ] **Request logging**: No audit trail for B2B compliance
- [ ] **WebSocket support**: No real-time streaming for conversation coach

### Required Enhancements

#### 1. Production API Key Management
```python
# Replace hardcoded dict with database-backed storage
class APIKeyManager:
    def __init__(self, db_path='api_keys.db'):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                owner TEXT,
                tier TEXT,  # free/pro/enterprise
                requests_today INTEGER DEFAULT 0,
                daily_limit INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def check_rate_limit(self, api_key: str) -> bool:
        # Check against persistent storage
        pass
    
    def increment_usage(self, api_key: str):
        # Track usage for billing
        pass
```

#### 2. Enhanced /explain with Context Attribution
```python
@app.post("/explain")
async def explain_with_context(text: str, context: Optional[List[str]] = []):
    """Return prediction with token attribution AND context influence"""
    
    # Get prediction with attributions
    result = run_inference(text, context)
    token_attrs = get_token_attributions(text, result['emotion'])
    
    # Analyze context influence
    context_influence = []
    if context:
        for i, ctx_turn in enumerate(context[-2:]):
            # Run inference with and without this context turn
            with_ctx = run_inference(text, context[:i+1])
            without_ctx = run_inference(text, context[:i])
            
            # Measure how much this turn changed the prediction
            influence = compute_prediction_shift(with_ctx, without_ctx)
            context_influence.append({
                'turn_index': i,
                'text': ctx_turn,
                'influence_score': influence,
                'key_phrases': extract_influential_phrases(ctx_turn)
            })
    
    return {
        **result,
        'token_attributions': token_attrs,
        'context_influence': context_influence,
        'method': 'integrated_gradients'
    }
```

---

## Surface 4: Conversation Coach (Consumer Product)

### ✅ Implemented
- [x] Core concept documented
- [x] Sarcasm-as-tone-checker use case defined

### ❌ Missing
- [ ] **Mobile-first UI**: No mobile app or PWA
- [ ] **Pre-send analysis**: No keyboard extension or share sheet integration
- [ ] **Reframe suggestions**: No LLM integration for alternative phrasings
- [ ] **Post-conversation replay**: No emotion arc visualization for past conversations
- [ ] **HITL feedback loop**: User corrections don't feed back to training pipeline automatically
- [ ] **Floating emotion meter**: No overlay widget

### Required Architecture
```
conversation_coach/
├── mobile_app/           # React Native or Flutter
│   ├── screens/
│   │   ├── PreSendScreen.js
│   │   ├── ArcReplayScreen.js
│   │   └── SettingsScreen.js
│   ├── components/
│   │   ├── EmotionMeter.js
│   │   ├── ReframeSuggestions.js
│   │   └── ArcTimeline.js
│   └── services/
│       ├── inference.js  # API client
│       └── feedback.js   # HITL submission
├── keyboard_extension/   # iOS/Android custom keyboard
│   └── ...
└── backend/
    └── reframe_engine.py # LLM-powered rewrite suggestions
```

### Reframe Engine Example
```python
# backend/reframe_engine.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ReframeEngine:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    
    def suggest_reframe(self, text: str, target_emotion: str) -> List[str]:
        """Generate 3 alternative phrasings with different emotional register"""
        prompt = f"Rewrite this message to express {target_emotion} instead: {text}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        outputs = self.model.generate(**inputs, max_length=128, num_return_sequences=3)
        
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
```

---

## Surface 5: Annotation Workspace (HITL Data Engine)

### ✅ Implemented
- [x] `/correct` endpoint logs corrections to JSONL
- [x] Contrastive pair verifier CLI exists (Stage 3)

### ❌ Missing
- [ ] **Web-based annotation UI**: No Streamlit or standalone tool
- [ ] **Real-time IAA calculation**: No multi-annotator agreement tracking
- [ ] **Hard cases queue**: No automatic flagging of ambiguous samples
- [ ] **Annotation guidelines**: No structured labeling interface
- [ ] **Annotator dashboards**: No productivity tracking
- [ ] **Export for retraining**: No automated pipeline from corrections to training data

### Required Implementation
```python
# annotation_workspace/app.py (Streamlit)
import streamlit as st
import json
from pathlib import Path

st.title("🏷️ Plutchik Annotation Workspace")

# Load queued samples
queue_file = Path('/workspace/data/hitl_queue.jsonl')
samples = [json.loads(line) for line in queue_file.read_text().strip().split('\n')]

# Show sample with model prediction
sample = st.session_state.get('current_sample', samples[0])

st.write(f"**Text**: {sample['text']}")
st.write(f"**Model Prediction**: {sample['predicted_emotion']} ({sample['confidence']:.2f})")
st.write(f"**Sarcasm**: {'Yes' if sample['predicted_sarcasm'] else 'No'}")

# Annotation form
with st.form("annotation_form"):
    true_emotion = st.selectbox("True Emotion", EMOTION_CLASSES)
    true_intensity = st.radio("Intensity", ['mild', 'primary', 'intense'])
    true_sarcasm = st.checkbox("Sarcastic?")
    confidence = st.slider("Your Confidence", 0.0, 1.0, 0.8)
    
    submitted = st.form_submit_button("Submit Annotation")
    
    if submitted:
        # Save annotation
        annotation = {
            **sample,
            'true_emotion': true_emotion,
            'true_intensity': true_intensity,
            'true_sarcasm': true_sarcasm,
            'annotator_id': st.session_state.get('annotator_id', 'anon'),
            'annotator_confidence': confidence,
            'timestamp': time.time()
        }
        
        # Append to annotations file
        with open('/workspace/data/annotations.jsonl', 'a') as f:
            f.write(json.dumps(annotation) + '\n')
        
        # Calculate IAA if second annotator
        existing = find_existing_annotation(sample['id'])
        if existing:
            iaa = compute_iaa(existing, annotation)
            if iaa < 0.6:
                st.warning(f"Low IAA ({iaa:.2f}) - flagged as hard case")
                # Move to hard_cases queue
```

---

## Surface 6: SDK / Embed

### ✅ Implemented
- [x] Concept documented

### ❌ Missing
- [ ] **React component**: No `<PluTchikWheel />` component
- [ ] **Vanilla JS embed**: No script tag solution
- [ ] **npm package**: No published package
- [ ] **PyPI package**: No `pip install plutchik-erc`
- [ ] **Explainability package**: No standalone Captum heatmap component

### Required Package Structure
```
plutchik-sdk/
├── packages/
│   ├── react-component/
│   │   ├── src/
│   │   │   ├── PluTchikWheel.tsx
│   │   │   ├── EmotionRadar.tsx
│   │   │   └── SarcasmIndicator.tsx
│   │   ├── package.json
│   │   └── README.md
│   ├── vanilla-js/
│   │   ├── dist/
│   │   │   └── plutchik-widget.js
│   │   └── examples/
│   │       └── index.html
│   └── python-client/
│       ├── plutchik_erc/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── visualize.py
│       └── setup.py
```

### React Component Example
```tsx
// packages/react-component/src/PluTchikWheel.tsx
import React, { useEffect, useState } from 'react';
import { Radar } from 'react-chartjs-2';

interface PluTchikWheelProps {
  text: string;
  apiUrl?: string;
  apiKey?: string;
  onPrediction?: (result: PredictionResult) => void;
}

export const PluTchikWheel: React.FC<PluTchikWheelProps> = ({
  text,
  apiUrl = 'http://localhost:8000',
  apiKey,
  onPrediction
}) => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const analyze = async () => {
      setLoading(true);
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey && { 'X-API-Key': apiKey })
        },
        body: JSON.stringify({ text })
      });
      const result = await response.json();
      setPrediction(result);
      onPrediction?.(result);
      setLoading(false);
    };
    analyze();
  }, [text]);

  if (loading) return <div>Analyzing...</div>;
  if (!prediction) return null;

  return (
    <div className="plutchik-wheel">
      <Radar data={buildRadarData(prediction.all_emotions)} />
      <SarcasmIndicator score={prediction.sarcasm_score} />
      <IntensityGauge intensity={prediction.intensity} />
    </div>
  );
};
```

---

## Surface 7: Content Moderation (B2B Trust & Safety)

### ✅ Implemented
- [x] Sarcasm detection working
- [x] Dyadic emotion detection (contempt, loathing, etc.)

### ❌ Missing
- [ ] **Escalation detection**: No thread-level emotion shift monitoring
- [ ] **Passive-aggression classifier**: No combined sarcasm+contempt signal
- [ ] **Human review queue**: No integration with moderation workflows
- [ ] **Bias audit dashboard**: No demographic bias tracking
- [ ] **Platform integrations**: No Discord/Twitter/Reddit API connectors

### Required Feature
```python
# content_moderation/escalation_detector.py
class ThreadEscalationDetector:
    def __init__(self, api_client):
        self.api = api_client
    
    def analyze_thread(self, comments: List[Comment]) -> EscalationReport:
        """Detect if a thread is escalating toward toxicity"""
        emotion_sequence = []
        
        for comment in sorted(comments, key=lambda c: c.timestamp):
            result = self.api.predict(comment.text)
            emotion_sequence.append({
                'turn': comment.turn_number,
                'emotion': result.emotion,
                'intensity_ring': result.primary_emotion_ring,
                'sarcasm': result.sarcasm
            })
        
        # Detect escalation pattern
        escalation_signals = []
        for i in range(len(emotion_sequence) - 1):
            prev = emotion_sequence[i]
            curr = emotion_sequence[i + 1]
            
            # Mild → Primary → Intense progression
            if self.is_escalating(prev, curr):
                escalation_signals.append({
                    'from_turn': prev['turn'],
                    'to_turn': curr['turn'],
                    'pattern': f"{prev['emotion']} → {curr['emotion']}"
                })
            
            # Sarcasm spike
            if curr['sarcasm'] and not prev['sarcasm']:
                escalation_signals.append({
                    'type': 'sarcasm_emergence',
                    'turn': curr['turn']
                })
        
        risk_score = len(escalation_signals) / len(emotion_sequence)
        
        return EscalationReport(
            risk_score=risk_score,
            signals=escalation_signals,
            recommendation='review' if risk_score > 0.5 else 'approve'
        )
```

---

## Surface 8: Customer Service Analytics (B2B SaaS)

### ✅ Implemented
- [x] Per-turn emotion labeling possible via `/predict/arc`

### ❌ Missing
- [ ] **Zendesk integration**: No webhook or API connector
- [ ] **Intercom integration**: No real-time agent coaching
- [ ] **CSAT correlation**: No analysis linking emotion arcs to satisfaction scores
- [ ] **Agent emotion tracking**: No separate analysis of agent messages
- [ ] **Intervention recommendations**: No suggestions for de-escalation
- [ ] **Zapier integration**: No no-code workflow connector

### Required Integration
```python
# customer_service/zendesk_connector.py
class ZendeskEmotionAnalytics:
    def __init__(self, zendesk_api_key, plutchik_api_key):
        self.zendesk = ZendeskClient(zendesk_api_key)
        self.plutchik = PlutchikClient(plutchik_api_key)
    
    def analyze_ticket(self, ticket_id: str) -> TicketEmotionReport:
        """Analyze emotion trajectory of a support ticket"""
        ticket = self.zendesk.get_ticket(ticket_id)
        comments = self.zendesk.get_comments(ticket_id)
        
        # Separate by speaker role
        customer_turns = [c.text for c in comments if c.author == ticket.requester]
        agent_turns = [c.text for c in comments if c.author == ticket.assignee]
        
        # Analyze each
        customer_arc = self.plutchik.predict_arc(customer_turns)
        agent_arc = self.plutchik.predict_arc(agent_turns)
        
        # Find critical moments
        critical_moments = []
        for turn in customer_arc['trajectory']:
            if turn['emotion'] in ['rage', 'loathing', 'contempt']:
                critical_moments.append({
                    'turn': turn['turn'],
                    'emotion': turn['emotion'],
                    'preceding_agent_message': agent_turns[turn['turn'] - 1] 
                        if turn['turn'] > 0 else None
                })
        
        return TicketEmotionReport(
            ticket_id=ticket_id,
            customer_arc=customer_arc,
            agent_arc=agent_arc,
            critical_moments=critical_moments,
            resolution_correlation=self.predict_resolution_likelihood(customer_arc)
        )
```

---

## Surface 9: Mental Health Support (Clinical Use Case)

### ✅ Implemented
- [x] 32-class detection including clinically relevant dyadic emotions
- [x] Longitudinal tracking possible via session IDs

### ❌ Missing
- [ ] **Journaling app**: No dedicated mobile/web app
- [ ] **Therapist dashboard**: No clinician-facing view
- [ ] **Trend analysis**: No 14-day rolling emotion profiles
- [ ] **Sarcasm rate tracking**: No correlation with avoidance coping
- [ ] **Crisis detection**: No suicide risk flagging (requires clinical validation)
- [ ] **HIPAA compliance**: No encryption, audit logs, or BAAs

### Ethical Considerations
⚠️ **This surface requires extreme caution:**
- Not a diagnostic tool (legal liability)
- Requires IRB approval for research use
- Needs clinical validation studies
- Must have crisis resource referrals
- Requires explicit informed consent

### Minimal Viable Implementation
```python
# mental_health/journal_analyzer.py
class JournalEmotionTracker:
    def __init__(self, plutchik_api):
        self.api = plutchik_api
    
    def analyze_entry(self, entry_text: str, entry_date: datetime) -> EntryAnalysis:
        result = self.api.predict(entry_text)
        return EntryAnalysis(
            date=entry_date,
            primary_emotion=result.emotion,
            sarcasm_present=result.sarcasm,
            intensity=result.intensity,
            all_emotions=result.all_emotions
        )
    
    def generate_trend_report(self, entries: List[EntryAnalysis], days: int = 14) -> TrendReport:
        """Generate 14-day rolling emotion profile"""
        recent = [e for e in entries if (datetime.now() - e.date).days <= days]
        
        # Aggregate emotion distributions
        avg_emotions = {}
        for emotion in EMOTION_CLASSES:
            avg_emotions[emotion] = np.mean([e.all_emotions.get(emotion, 0) for e in recent])
        
        # Top 5 emotions
        top_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Sarcasm rate trend
        sarcasm_rate = sum(1 for e in recent if e.sarcasm_present) / len(recent)
        
        # Detect significant shifts
        if len(recent) >= 7:
            week1 = recent[:len(recent)//2]
            week2 = recent[len(recent)//2:]
            grief_week1 = np.mean([e.all_emotions.get('grief', 0) for e in week1])
            grief_week2 = np.mean([e.all_emotions.get('grief', 0) for e in week2])
            grief_change = grief_week2 - grief_week1
        else:
            grief_change = 0
        
        return TrendReport(
            period_days=days,
            entry_count=len(recent),
            top_emotions=top_emotions,
            sarcasm_rate=sarcasm_rate,
            grief_trend=grief_change,
            interpretation=self._generate_interpretation(top_emotions, grief_change)
        )
```

---

## Surface 10: Creative Writing Tools

### ✅ Implemented
- [x] Character voice analysis theoretically possible

### ❌ Missing
- [ ] **VS Code extension**: No IDE integration
- [ ] **Scrivener plugin**: No writing software integration
- [ ] **Character emotion profile**: No aggregation by character name
- [ ] **Scene arc visualization**: No chapter/scene-level breakdown
- [ ] **Sarcasm density metric**: No corpus-level style analysis
- [ ] **Consistency alerts**: No flagging of out-of-character emotions

### VS Code Extension Concept
```typescript
// creative-writing/vscode-extension/src/extension.ts
export function activate(context: vscode.ExtensionContext) {
    const analyzeCommand = vscode.commands.registerCommand(
        'plutchik.analyzeCharacter',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;
            
            const text = editor.document.getText();
            const characters = extractCharacterDialogues(text);
            
            const results = await Promise.all(
                Object.entries(characters).map(async ([name, utterances]) => {
                    const emotions = await plutchikApi.predictBatch(utterances);
                    return {
                        character: name,
                        emotionDistribution: aggregateEmotions(emotions),
                        sarcasmRate: calculateSarcasmRate(emotions),
                        consistencyScore: calculateConsistency(emotions)
                    };
                })
            );
            
            // Show in webview panel
            const panel = vscode.window.createWebviewPanel(
                'plutchikAnalysis',
                'Character Emotion Analysis',
                vscode.ViewColumn.Beside
            );
            panel.webview.html = renderAnalysisHTML(results);
        }
    );
    
    context.subscriptions.push(analyzeCommand);
}
```

---

## Surface 11: Negotiation Coaching (Enterprise EQ Training)

### ✅ Implemented
- [x] Dyadic emotion detection (submission, aggressiveness, contempt, awe)

### ❌ Missing
- [ ] **Pre-negotiation analyzer**: No message pre-flight checking
- [ ] **Post-negotiation replay**: No annotated conversation transcript
- [ ] **Trust signal tracking**: No specific monitoring of trust-ring emotions
- [ ] **Real-time coaching**: No live suggestions during negotiation
- [ ] **Enterprise SSO**: No Okta/Auth0 integration
- [ ] **Compliance logging**: No audit trail for regulated industries

---

## Surface 12: Media Bias Analysis (Research/Social Science)

### ✅ Implemented
- [x] Emotion framing detection theoretically possible

### ❌ Missing
- [ ] **Longitudinal tracker**: No time-series emotion profiling
- [ ] **Political speech corpus**: No pre-loaded dataset
- [ ] **Cross-source comparison**: No side-by-side outlet analysis
- [ ] **Framing bias metrics**: No quantitative bias scores
- [ ] **Academic export**: No paper-ready statistical outputs
- [ ] **Bias audit for model itself**: No self-bias-detection

---

## Phase 8 Extensions (Architectural Expansions)

### Extension 1: Emotion Arc Prediction (GRU Sequence Model)
**Status**: ❌ Not started  
**Priority**: HIGH - Most direct Phase 8 project  
**Implementation Effort**: 2-3 days  

```python
# models/emotion_arc_predictor.py
class EmotionArcPredictor(nn.Module):
    def __init__(self, encoder_output_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=encoder_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32)  # 32 emotion classes
        )
    
    def forward(self, cls_embeddings_sequence, lengths=None):
        """
        Args:
            cls_embeddings_sequence: (batch, seq_len, 128) - CLS embeddings from turns 0..T
        Returns:
            predicted_emotion_dist: (batch, 32) - Predicted distribution for turn T+1
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            cls_embeddings_sequence, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        # Use last hidden state
        output = self.prediction_head(hidden[-1])
        return output
    
    def predict_inflection(self, current_dist, predicted_next_dist):
        """Detect if KL divergence indicates emotional turning point"""
        kl_div = F.kl_div(
            F.log_softmax(predicted_next_dist, dim=-1),
            F.softmax(current_dist, dim=-1),
            reduction='batchmean'
        )
        return kl_div > 0.5  # Threshold for inflection
```

**Training Data**: Already exists in `train.jsonl` - use turn T+1 emotion as label for sequence 0..T

---

### Extension 2: Speaker Profiling (Persistent Embeddings)
**Status**: ❌ Not started  
**Priority**: MEDIUM - Privacy concerns require careful design  
**Implementation Effort**: 3-5 days  

```python
# models/speaker_aware_model.py
class SpeakerAwarePluTchikModel(PluTchikMultiTaskModel):
    def __init__(self, num_speakers=1000, speaker_embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        # Speaker embedding table
        self.speaker_embeddings = nn.Embedding(num_speakers, speaker_embed_dim)
        # Projection to match hidden size
        self.speaker_projection = nn.Linear(speaker_embed_dim, 128)
        # Update emotion classifier to accept concatenated input
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 128, 128),  # pooled + speaker
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32)
        )
    
    def forward(self, input_ids, attention_mask, speaker_ids=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        if speaker_ids is not None:
            speaker_embed = self.speaker_embeddings(speaker_ids)
            speaker_proj = self.speaker_projection(speaker_embed)
            # Concatenate with pooled output
            combined = torch.cat([pooled_output, speaker_proj], dim=-1)
            emotion_logits = self.emotion_classifier(combined)
        else:
            emotion_logits = self.emotion_classifier(pooled_output)
        
        # Rest of forward pass same as parent class
        ...
```

**Privacy Requirements**:
- Opt-in only
- Local storage (IndexedDB in browser)
- Deletable on request
- Never transmitted to server

---

### Extension 3: Multimodal Fusion (Text + Audio + Video)
**Status**: ❌ Not started  
**Priority**: LOW - Phase 9 problem  
**Implementation Effort**: 2-3 weeks  

**Recommended First Step**: Audio-only fusion (text + wav2vec2)

```python
# models/multimodal_fusion.py
class MultimodalEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Text encoder (existing)
        self.text_encoder = PluTchikMultiTaskModel()
        # Audio encoder (pretrained wav2vec2)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # Cross-attention fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.3
        )
        # Modality gating network
        self.modality_gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, audio_waveforms=None):
        # Text encoding
        text_outputs = self.text_encoder.bert(input_ids, attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        
        # Audio encoding (if available)
        if audio_waveforms is not None:
            audio_outputs = self.audio_encoder(audio_waveforms)
            audio_cls = audio_outputs.last_hidden_state[:, 0, :]
            audio_proj = self.audio_projection(audio_cls)
            
            # Cross-attention fusion
            fused, _ = self.fusion_layer(
                text_cls.unsqueeze(0),
                audio_proj.unsqueeze(0),
                audio_proj.unsqueeze(0)
            )
            fused = fused.squeeze(0)
            
            # Gating: learn which modality to trust
            gate = self.modality_gate(torch.cat([text_cls, audio_proj], dim=-1))
            final_rep = gate * fused + (1 - gate) * text_cls
        else:
            final_rep = text_cls
        
        # Task heads
        emotion_logits = self.emotion_classifier(final_rep)
        ...
```

---

### Extension 4: Emotion-Conditioned Generation (RLHF-style)
**Status**: ❌ Not started  
**Priority**: MEDIUM - Breaks contrastive pair generation bottleneck  
**Implementation Effort**: 1-2 weeks  

```python
# models/emotion_conditioned_generator.py
class EmotionConditionedGenerator:
    def __init__(self, generator_name="gpt2", discriminator=None):
        self.generator = AutoModelForCausalLM.from_pretrained(generator_name)
        self.discriminator = discriminator or PluTchikMultiTaskModel()
        self.tokenizer = AutoTokenizer.from_pretrained(generator_name)
    
    def generate_with_emotion_constraint(
        self,
        prompt: str,
        target_emotion: str,
        target_intensity: str = 'primary',
        target_sarcasm: bool = False,
        num_candidates: int = 5
    ) -> List[GeneratedCandidate]:
        """Generate text that matches target emotional profile"""
        candidates = []
        
        for _ in range(num_candidates):
            # Generate candidate
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.generator.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                temperature=0.8
            )
            candidate_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Score with discriminator
            disc_result = self.discriminator.predict(candidate_text)
            
            # Calculate reward
            emotion_reward = disc_result.all_emotions[target_emotion]
            intensity_reward = 1.0 if disc_result.intensity == target_intensity else 0.5
            sarcasm_reward = 1.0 if disc_result.sarcasm == target_sarcasm else 0.5
            
            total_reward = (emotion_reward + intensity_reward + sarcasm_reward) / 3
            
            candidates.append(GeneratedCandidate(
                text=candidate_text,
                reward=total_reward,
                emotion_match=emotion_reward,
                intensity_match=intensity_reward,
                sarcasm_match=sarcasm_reward
            ))
        
        return sorted(candidates, key=lambda c: c.reward, reverse=True)
```

---

### Extension 5: Plutchik-as-Evaluation (LLM Output Evaluator)
**Status**: ❌ Not started  
**Priority**: HIGH - Positions model as LLM infrastructure  
**Implementation Effort**: 3-5 days  

```python
# evaluation/llm_emotion_evaluator.py
class LLMEmotionEvaluator:
    def __init__(self, plutchik_api):
        self.api = plutchik_api
    
    def evaluate_story_emotional_arc(
        self,
        story_text: str,
        target_arc: List[str],
        segment_by: str = 'paragraph'
    ) -> EmotionCoherenceScore:
        """
        Evaluate whether generated story achieves target emotion trajectory.
        
        Args:
            story_text: Full generated story
            target_arc: List of target emotions for each segment, e.g.,
                       ['trust', 'grief', 'acceptance']
            segment_by: How to divide story ('paragraph', 'chapter', 'sentence')
        
        Returns:
            Coherence score 0-1 indicating how well story matches target arc
        """
        segments = self._segment_story(story_text, segment_by)
        
        # Get emotion distribution for each segment
        segment_emotions = []
        for segment in segments:
            result = self.api.predict(segment)
            segment_emotions.append(result.all_emotions)
        
        # Compare to target arc
        alignment_scores = []
        for i, (segment_dist, target_emotion) in enumerate(zip(segment_emotions, target_arc)):
            target_prob = segment_dist.get(target_emotion, 0)
            alignment_scores.append(target_prob)
        
        # Penalize for staying in mild ring throughout
        intensity_variety = self._calculate_intensity_variety(segments)
        
        overall_score = np.mean(alignment_scores) * intensity_variety
        
        return EmotionCoherenceScore(
            overall=overall_score,
            segment_alignment=alignment_scores,
            intensity_variety=intensity_variety,
            recommendations=self._generate_recommendations(segment_emotions, target_arc)
        )
```

---

### Extension 6: Cross-Lingual Transfer (Multilingual Adaptation)
**Status**: ❌ Not started  
**Priority**: MEDIUM - Research value high, engineering lift moderate  
**Implementation Effort**: 1-2 weeks  

```python
# models/multilingual_plutchik.py
class MultilingualPluTchikModel(PluTchikMultiTaskModel):
    def __init__(self, base_model_name="xlm-roberta-base", **kwargs):
        # Use XLM-RoBERTa instead of BERT
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # Task heads remain language-agnostic
        super().__init__(
            encoder=self.encoder,
            config=self.config,
            **kwargs
        )
    
    @classmethod
    def adapt_from_english(cls, english_checkpoint, target_language):
        """
        Adapt English-trained model to target language.
        
        Strategy:
        1. Load English weights for task heads
        2. Initialize encoder with XLM-R pretrained weights
        3. Fine-tune on translated/augmented data for target language
        """
        model = cls()
        
        # Load task head weights from English model
        english_model = PluTchikMultiTaskModel()
        english_model.load_state_dict(torch.load(english_checkpoint)['model_state_dict'])
        
        model.emotion_classifier.load_state_dict(english_model.emotion_classifier.state_dict())
        model.sarcasm_head.load_state_dict(english_model.sarcasm_head.state_dict())
        model.intensity_head.load_state_dict(english_model.intensity_head.state_dict())
        
        # Encoder remains with XLM-R initialization (zero-shot transfer)
        # Or fine-tune on translated data
        
        return model
```

**Priority Order for Languages**:
1. Spanish (highest overlap with English training data)
2. Mandarin (most linguistically distant, highest research value)
3. Hindi (large speaker population, underrepresented in NLP)
4. Arabic (right-to-left, morphologically rich)

---

## Critical Path Recommendations

### Week 1: Foundation Fixes
1. **Fix model path** in `inference_server.py` (10 minutes)
2. **Add proper Captum integration** to `/explain` endpoint (4 hours)
3. **Create comparative analysis endpoint** `/compare` (2 hours)
4. **Build annotation workspace UI** (Streamlit, 1 day)

### Week 2: Phase 8 Extensions
1. **Implement emotion arc prediction GRU** (2 days)
2. **Train on existing dialogue sequences** (4 hours)
3. **Deploy as new endpoint** `/predict/next` (2 hours)
4. **Write evaluation benchmarks** (4 hours)

### Week 3: Browser Plugin MVP
1. **Export model to ONNX** (4 hours)
2. **Create Chrome extension skeleton** (4 hours)
3. **Implement Reddit overlay** (1 day)
4. **Test with real Reddit threads** (4 hours)

### Week 4: B2B Readiness
1. **Add production API key management** (SQLite-backed, 4 hours)
2. **Implement usage analytics** (2 hours)
3. **Add request logging** (2 hours)
4. **Create Zapier integration prototype** (1 day)

---

## Dataset Moat Analysis

### Current Strengths
✅ **32-class resolution**: Order of magnitude more granular than MELD/EmoryNLP (6-7 classes)  
✅ **IAA scores per utterance**: Continuous uncertainty signal for loss weighting  
✅ **Sarcasm co-annotation**: Same utterances labeled for both emotion + sarcasm  
✅ **Emotion cause metadata**: Rare signal enabling cause-aware analysis  
✅ **Dialogue structure preserved**: Speaker roles, turn order, conversation IDs  

### Current Weaknesses
❌ **Synthetic monoculture**: 266 handcrafted dialogues from single author  
❌ **Near-duplicates exist**: `disapproval_policy` appears twice (noted in vision doc)  
❌ **Inconsistent cause field length**: 3 words to 10 words - limits downstream use  
❌ **Only 1,200 utterances**: Small for deep learning (but high quality)  
❌ **English-only**: Cultural bias embedded in emotion expression  

### Moat-Growth Mechanisms
1. **HITL pipeline**: Every user correction → labeled sample (2 seconds, zero cost)
2. **Contrastive pairs**: Context-swapped sarcasm twins (unique dataset contribution)
3. **Per-scenario sarcasm rates**: Statistical characterization (workplace 47%, romance 9%)

### Required Data Infrastructure
```python
# data/deduplication_pipeline.py
def detect_near_duplicates(dataset: List[Dict], threshold=0.95):
    """Remove near-duplicate dialogues"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [d['text'] for d in dataset]
    embeddings = model.encode(texts)
    
    # Compute pairwise cosine similarity
    similarities = cosine_similarity(embeddings)
    
    duplicates = []
    for i in range(len(dataset)):
        for j in range(i+1, len(dataset)):
            if similarities[i,j] > threshold:
                duplicates.append((i, j, similarities[i,j]))
    
    # Keep higher-quality version (higher IAA score)
    to_remove = set()
    for i, j, sim in duplicates:
        if dataset[i]['iaa_score'] < dataset[j]['iaa_score']:
            to_remove.add(i)
        else:
            to_remove.add(j)
    
    cleaned_dataset = [d for idx, d in enumerate(dataset) if idx not in to_remove]
    
    return cleaned_dataset, duplicates
```

---

## Key Tensions to Navigate

| Tension | Trade-off | Recommended Resolution |
|---------|-----------|----------------------|
| Expressiveness (32 classes) vs Reliability (fewer classes = higher IAA) | More nuance vs more agreement | Keep 32 classes, use IAA-weighted loss to downweight ambiguous samples |
| On-device inference (privacy) vs Model quality (larger = better) | Privacy vs F1 score | Distill to TinyBERT for browser, keep full model for API |
| Confidence transparency vs User experience | Uncertainty visibility vs definitive answers | Show confidence when < 0.55, hide when > 0.8 |
| Sarcasm detection accuracy vs Emotion signal integrity (GRL tradeoff) | Better sarcasm vs cleaner emotion features | Use Gradient Reversal Layer with tunable lambda coefficient |

---

## Success Metrics

### Technical Metrics
- Macro-F1 (32-class): Target ≥ 0.75 (current ~0.71)
- Sarcasm AUC: Target ≥ 0.88 (current ~0.84)
- Intensity Accuracy: Target ≥ 0.82 (current ~0.78)
- Inference Latency (CPU): Target ≤ 15ms (current ~12ms ✓)
- Model Size: Target ≤ 10MB for browser (current 50MB PT, need ONNX)

### Product Metrics
- Daily Active Users (Dashboard): Target 100+ within 3 months
- API Calls/Month (B2B): Target 10,000+ within 6 months
- Browser Extension Installs: Target 1,000+ within 3 months
- HITL Corrections Submitted: Target 500+/month (data moat growth)
- Annotation IAA: Target ≥ 0.75 average across annotators

### Business Metrics
- B2B Customers (paid API): Target 5+ within 6 months
- Mental Health Partnerships: Target 2 pilot programs within 9 months
- Research Citations: Target 3+ papers using Plutchik-as-Evaluation
- Open Source Stars: Target 500+ GitHub stars within 3 months

---

## Conclusion

**What's Working Well:**
- Core model architecture is sound (multi-task, 32-class, sarcasm, intensity)
- API endpoints cover 80% of use cases
- Dashboard provides good visualization foundation
- Dataset has unique strengths (IAA scores, sarcasm co-annotation, causes)

**Critical Gaps:**
- No real Captum explainability (just placeholder)
- No browser plugin code (only architecture docs)
- No annotation workspace UI (only JSONL logging)
- No speaker profiling or arc prediction (Phase 8 extensions)
- Model path broken (won't load without fix)

**Highest-ROI Next Steps:**
1. Fix model path (10 min)
2. Add Captum to `/explain` (4 hrs)
3. Build annotation UI (1 day)
4. Implement GRU arc predictor (2 days)
5. Export ONNX + create Chrome extension skeleton (2 days)

**Strategic Insight:** The dataset is the moat, not the model. Every design decision should maximize HITL data collection. The annotation workspace is not a nice-to-have—it's the data engine that compounds value over time.
