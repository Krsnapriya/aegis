# Advanced Dynamic Analysis API

This endpoint provides **real-time emotional intelligence** beyond simple classification. It uses Neural ODEs for trajectory forecasting, multimodal incongruity detection for sarcasm, and counterfactual generation for constructive reframing.

## Endpoint: `POST /analyze/dynamic`

### Request
```json
{
  "text": "Oh GREAT, another meeting! Just what I needed!!!",
  "session_id": "user-123",
  "user_baseline": {
    "average_vector": [0.1, 0.1, ...]  // Optional: user's typical emotion profile
  }
}
```

### Response
```json
{
  "risk_level": "high",
  "sarcasm_probability": 0.85,
  "trajectory_forecast": [[0.1, 0.2, ...], ...],  // 10-step prediction
  "inflection_point": 3,
  "reframe_suggestions": [
    "I understand, perhaps we can look at this differently.",
    "I hear you. Let's consider the options.",
    "Maybe we can approach this from another angle."
  ],
  "baseline_deviation": {
    "distance": 0.52,
    "message": "This tone deviates significantly from your usual style (Deviation: 0.52)"
  },
  "signals": [
    "Positive words with aggressive capitalization/punctuation"
  ]
}
```

## Key Features

### 1. Neural ODE Forecasting
- Models emotional state as a **continuous dynamical system**
- Predicts future emotional trajectory over next 10 time steps
- Detects **inflection points** where emotional acceleration is highest
- Risk score based on average acceleration magnitude

### 2. Multimodal Incongruity Detection
- Compares semantic sentiment vs pragmatic markers (caps, punctuation, emoji)
- Detects passive-aggression that keyword filters miss
- Returns specific signals explaining why sarcasm was detected

### 3. Counterfactual Reframing
- Generates 3 alternative phrasings optimized for de-escalation
- Targets trust/serenity registers when risk is high
- Preserves core message while adjusting emotional register

### 4. Personalized Baseline Comparison
- Tracks user's historical emotion distribution
- Flags significant deviations from personal norm
- Enables coaching like "This is unusually aggressive for you"

## Use Cases

### Pre-Send Warning (Browser Extension)
```javascript
const analysis = await fetch('/analyze/dynamic', {
  text: draftText,
  session_id: currentUser
});

if (analysis.risk_level === 'high') {
  showWarning(analysis.reframe_suggestions);
  blockSend();
}
```

### Live Thread Monitoring
```javascript
// Poll every 30 seconds during active conversation
setInterval(async () => {
  const forecast = await getTrajectoryForecast(sessionId);
  if (forecast.max_acceleration > threshold) {
    alert('Conversation escalating - inflection point predicted in 3 turns');
  }
}, 30000);
```

### Coaching Dashboard
```python
# Show user their emotional arc over time
deviation = response.baseline_deviation
if deviation:
    display(f"⚠️ {deviation['message']}")
    display("Consider:", response.reframe_suggestions[0])
```

## Algorithm Details

### Neural ODE Architecture
- **Latent dim**: 64
- **Context dim**: 128 (from dialogue history pooling)
- **Integration method**: RK4 (Runge-Kutta 4th order)
- **Time horizon**: 1.0 units (normalized conversation time)

### Incongruity Features
1. Punctuation aggression (`!`, `?` count)
2. Capitalization ratio
3. Lexical contrast (positive words vs negative context)
4. Response length vs sentiment mismatch

### Reframe Generation
- Template-based perturbations for MVP
- Future: T5/GPT fine-tuned on emotion-conditioned generation
- Targets: trust, serenity, acceptance for de-escalation
