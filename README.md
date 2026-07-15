---
title: Plutchik Emotion Engine
emoji: "\U0001F3AD"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Plutchik ERC Engine

[![Version: 2.5.1](https://img.shields.io/badge/Version-2.5.1-brightgreen.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)
[![Architecture: Multi-Task Transformer](https://img.shields.io/badge/Architecture-Multi--Task--Transformer-blue.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)
[![Taxonomy: Plutchik 32](https://img.shields.io/badge/Taxonomy-Plutchik--32-orange.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)

Multi-task RoBERTa model for Emotion Recognition in Conversation (ERC) using Plutchik's 32-class psychoevolutionary taxonomy. Predicts emotion, sarcasm, intensity, and domain invariance from a shared encoder, with Neural ODE trajectory forecasting and a human-in-the-loop correction flywheel.

---

## 1. Dataset

### 1.1 Production Corpus

`data/processed/ERC/plutchik_v2_production.csv` — 2,800 utterances across 1,346 dialogues.

| Split | Rows |
|---|---|
| Train | 2,040 |
| Val | 390 |
| Test | 370 |

Each row carries: `dialogue_id`, `turn_id`, `speaker`, `text`, `emotion` (32-class), `emotion_ring`, `scenario`, `sarcasm_flag`, `emotion_cause`, `sentiment_polarity`, `utterance_word_count`, `inter_annotator_agreement` (0.65–1.0, mean 0.825), `confidence_score`, `topic`, `emotion_shift`, pre-tokenized `input_ids`/`attention_mask`, and `split`.

- **Sarcasm prevalence**: 744/2,800 (26.6%)
- **Emotion shift**: 1,384/2,800 turns flagged as representing a shift from the previous emotion
- **Scenarios**: 14 environments (workplace, social, conflict, friendship, family, romance, support, academic, casual, travel, wellbeing, community, technology)
- **All 32 Plutchik classes** present in the data

### 1.2 Handcrafted vs. Synthetic

- **Handcrafted dialogues** (production corpus): Multi-turn conversations with speakers, explicit emotion causes, and human IAA scores. Average 6–8 turns per dialogue.
- **Synthetic data (excluded)**: `nuanced_generated` (985 rows, template-based single-turn) and `vast_expansion` (80 rows, mathematically padded). Both are filtered out at training time via `build_dataset_from_csv(exclude_synthetic=True)`. The original `scripts/augmenter.py` that generated these is deprecated.

### 1.3 Inter-Annotator Agreement

Each row carries an IAA score (0.65–1.0). The `validate_iaa_scores()` diagnostic checks whether scores are formula-derived (binary split correlated with sarcasm) or reflect genuine annotator disagreement. IAA scores are used as per-sample loss weights during training — samples with higher agreement contribute more to the gradient.

### 1.4 CDA Pipeline (Contrastive Data Augmentation)

A human-in-the-loop pipeline for generating contrastive dissonance pairs to train the Dissonance Head:

1. **Template generation** (`utils/augmenter.py`): Extracts sarcastic training samples and creates JSONL templates with the original text/context/emotion and placeholders for a human-authored "twin" (literal re-reading with a different context and emotion).
2. **Human verification** (`utils/pair_verifier.py`): An interactive CLI presents each template to a human annotator. The annotator fills in `twin_context` and `twin_emotion`, then passes a 3-gate verification: (a) does the twin context make the text read literally? (b) is the twin emotion plausible? (c) would a human agree without coaching?
3. **Training merge gate**: CDA pairs are merged into training only when the `PLUTCHIK_CDA_JSONL` env var points to a JSONL file with at least `PLUTCHIK_CDA_MIN_PAIRS` (default 200) human-verified rows.

---

## 2. Model Architecture

```
Input: [CONTEXT] turn_N-2 | turn_N-1 [/CONTEXT] [CURRENT] [SCENARIO] workplace [/SCENARIO]
       [TOPIC] termination [/TOPIC] utterance_text [/CURRENT]
                          |
                  RoBERTa-Base (12-layer, 768-hidden)
                          |
                    CLS token (768-d)
                          |
                   shared_dense (768->768, ReLU, Dropout, LayerNorm)
                     /     |       \            \
          emotion_head  sarcasm_head  intensity_head  scenario_discriminator
          (384->32)     (384->2)      (384->1,Sig)    (GRL -> 384->2)
```

For CDA batches, an additional dual-encoder path:

```
Context-only input -> RoBERTa (shared) -> CLS -> context_pooler (768->384, Tanh)
                                                    |
                                         cat(shared_rep, ctx_rep) -> dissonance_head (1152->192->1, Sigmoid)
```

### 2.1 Multi-Task Heads

| Task | Head | Output | Loss |
|---|---|---|---|
| Emotion Classification | 2-layer MLP (768->384->32) | 32-way logits | Cross-Entropy |
| Sarcasm Detection | 2-layer MLP (768->384->2) | Binary logits | Cross-Entropy |
| Intensity Regression | 2-layer MLP (768->384->1) | Scalar in [0,1] (sigmoid) | MSE |
| Adversarial Domain Discriminator | 2-layer MLP (768->384->2) | Binary (formal vs. social) | Cross-Entropy via GRL |

### 2.2 Gradient Reversal Layer (GRL)

`GradReverse` is a custom `torch.autograd.Function`. Forward pass: identity. Backward pass: negates gradients scaled by `alpha`. Forces the shared encoder to learn features invariant to scenario domain. Binary domain labels derived from sarcasm-rate analysis: high-sarcasm scenarios (workplace, social, conflict, casual, friendship, romance) = class 0; low-sarcasm = class 1.

### 2.3 Dissonance Detection Head (Stage 3)

Dual-encoder architecture: the main encoder processes the full `[CONTEXT]...[/CONTEXT] [CURRENT]...[/CURRENT]` input; a separate `context_pooler` (Linear + Tanh) processes context-only input through shared RoBERTa weights. Text representation (768-d) and context representation (384-d, projected) are concatenated and fed through a 3-layer MLP with final Sigmoid to produce a dissonance score in [0,1]. Only activated when `context_input_ids` are provided (CDA contrastive batches).

---

## 3. Algorithms

### 3.1 Wheel-Distance Metric

A custom Plutchik-specific distance measure between any two of the 32 emotions:

```
distance(e1, e2) = 0.85 * angular_distance / pi + 0.15 * ring_distance / 2
```

- **Angular distance**: Shortest path around the 8-sector wheel, normalized to [0,1]. Sectors are 45 degrees apart.
- **Ring distance**: Absolute difference in ring intensities (mild=0, primary=1, dyadic=1.5, intense=2), normalized by /2.
- Cross-sector errors (e.g., rage predicted instead of joy) are penalized roughly 5x more than same-sector intensity errors.
- A full 32x32 matrix (`WHEEL_DISTANCE_MATRIX`) is precomputed at import time.

**Wheel-distance loss scaling**: `exp(alpha * d(pred, true))` with alpha=2.0. Rage->joy (d=0.85) gets ~5.2x penalty; rage->annoyance (d=0.15) gets ~1.2x. Applied to per-sample emotion cross-entropy loss.

### 3.2 Neural ODEs for Emotion Trajectory Forecasting

`EmotionODEFunc` is a 3-layer MLP modeling dh/dt = f(h(t), t, context), where h is a 64-dim latent emotion state and context is a 128-dim conversation summary. `TrajectoryForecaster` encodes the 32-dim emotion vector into 64-dim latent space, solves the ODE using RK4 integration via `torchdiffeq.odeint`, then decodes back to 32-dim softmax probabilities. Computes velocity and acceleration to detect **inflection points** — moments where emotional trajectory changes direction.

> **Note**: The encoder/decoder are currently randomly initialized (simulation mode). The ODE structure is correct but magnitudes are uncalibrated.

### 3.3 Multimodal Incongruity Detector

Rule-based sarcasm/passive-aggression detector using 5 signal channels:

1. **Polarity contradiction**: Mixing positive and negative words (+0.3), or positive words with aggressive caps/exclamations (+0.4).
2. **Emphasis markers**: 3+ exclamation marks or 2+ with high caps ratio (+0.2).
3. **Intensifiers without substance**: 2+ intensifier words in a short utterance (<12 words) (+0.2).
4. **Passive-aggressive phrases**: Detection of phrases like "just what i needed", "oh great", "thanks a lot" (+0.4).
5. **Terse phrasing with elevated sentiment**: Fewer than 8 words with high semantic sentiment (+0.2).

Outputs sarcasm probability (capped at 0.95), list of signals, and passive-aggressive flag.

### 3.4 Counterfactual Generator / Strategic Reframe

Template-based text rewriting targeting specific Plutchik emotions: softens language for trust/serenity targets ("I understand", "Perhaps"), intensifies for anger targets ("Frankly", exclamation marks), reframes problems as opportunities for joy targets. Not a learned model — deterministic template logic.

### 3.5 Integrated Gradients (IG) for Explainability

Uses Captum's `LayerIntegratedGradients` on RoBERTa embeddings layer. Computes attribution scores for each token relative to the predicted emotion class. Uses a padding-token baseline. Normalizes attributions and splits them into context-span and current-span top tokens. Default step count: 10 (reduced from 50 for dashboard speed).

### 3.6 LLM Cross-Validation

`NemotronClient` calls NVIDIA Nemotron-3 (120B params) via OpenRouter API. Sends a structured prompt asking the LLM to analyze Plutchik emotion, sarcasm, intensity, and reasoning in JSON. Includes retry/timeout handling (30s), JSON extraction with fallback parsing, emotion validation against the canonical 32-class list, and a heuristic keyword-based mock fallback when the API is unavailable.

### 3.7 Input Sanitization

`InputSanitizer` provides:
- **Adversarial token stripping**: Removes reserved tokens like `[CONTEXT]`, `[/CURRENT]` to prevent prompt injection.
- **Emoji-only bypass**: Maps 30+ emojis directly to Plutchik emotions for emoji-only inputs.
- **Trigram gibberish detection**: Character-level n-gram log-probability model from a reference English corpus; rejects inputs below threshold.
- **Max length enforcement**: 5,000 character limit.

### 3.8 KL Divergence for Arc Analysis

The conversation arc endpoint uses KL divergence between consecutive emotion probability distributions to detect turning points. Divergence > mean + 1.5 * std flagged as a turning point. Intensity delta analysis classifies arcs as "stable", "escalation", "de-escalation", or "volatile".

### 3.9 Human-in-the-Loop Flywheel

`/correct` endpoint accepts predicted vs. corrected emotion pairs persisted to the database. When enough corrections accumulate (>= `min_samples`), `utils/flywheel_trainer.py` triggers fine-tuning: loads the current model, creates a temporary CSV from corrections (IAA weight=2.0), and runs 3 epochs at lr=1e-5.

---

## 4. Training

### 4.1 Configuration

| Parameter | Value |
|---|---|
| Batch size | 4 (reduced for MPS OOM on M1) |
| Epochs | 1 (fast prototype; production uses 5) |
| Learning rate | 5e-5 |
| Max sequence length | 128 tokens |
| Optimizer | AdamW |
| LR Scheduler | CosineAnnealingLR |
| Seed | 42 (fully deterministic: Python, NumPy, PyTorch, CUDA) |

### 4.2 Loss Function

```
L_total = 1.0 * L_emotion + 0.7 * L_sarcasm + 0.5 * L_intensity + 0.3 * L_adv + 1.0 * L_dissonance
```

- `L_emotion`: Cross-Entropy (32-way), scaled by wheel-distance weight and IAA weight
- `L_sarcasm`: Cross-Entropy (binary), scaled by IAA weight
- `L_intensity`: MSE (regression), scaled by IAA weight
- `L_adv`: Cross-Entropy (binary domain classification through GRL)
- `L_dissonance`: Binary Cross-Entropy (only active for CDA contrastive samples)

### 4.3 Adversarial Warmup

GRL lambda follows a sigmoid warmup: alpha = 0.0 for the first N epochs (warmup), then alpha = (2 / (1 + exp(-10 * p)) - 1) * grl_lambda_max where p is the fraction of remaining training steps. Default grl_lambda_max = 0.5.

### 4.4 Hardware & Mixed Precision

- Device auto-detection: CUDA -> MPS (Apple Silicon) -> CPU
- AMP enabled on CUDA via `torch.amp.autocast` and `GradScaler`
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- Best model selection: Validation F1-Macro (not accuracy, to handle class imbalance)
- Emotion centroids saved as pickle after each improvement for explainability

### 4.5 Benchmarks

| Metric | Value |
|---|---|
| Training Accuracy | 80.2% |
| Validation Accuracy | 54.1% |
| F1-Macro (Emotion) | 0.45 |
| Inference Latency (Standard) | ~250ms |
| Inference Latency (Fast Attribution) | ~800ms CPU / ~150ms GPU |
| Inference Latency (Full Captum) | ~15s CPU / ~2s GPU |

---

## 5. Deployment

### 5.1 Architecture

- **Inference Server** (`inference_server.py`): FastAPI on port 8000. Endpoints: `/predict`, `/explain`, `/predict/batch`, `/predict/arc`, `/analyze/dynamic`, `/correct`, `/corrections/stats`, `/emotions`, `/session/{id}`, `/health`, `/reload`.
- **Dashboard** (`app.py`): Streamlit with glassmorphism CSS. Five analysis modes: Single Utterance, Conversation Arc, Comparative Analysis, Dynamic Intelligence, Batch File Upload.
- **Database**: SQLAlchemy with PostgreSQL (primary) and SQLite (fallback). Single-writer background thread with exponential-backoff retry.
- **Security**: API key validation, rate limiting (30 req/min per IP), input sanitization, CORS restriction, Pydantic validation.

### 5.2 Session Management

Thread-safe `SessionManager` with RLock, sliding context window of 3 turns, emotion vector history (last 10 turns for ODE forecasting), LRU eviction at 1,000 sessions.

### 5.3 Environment Variables

| Variable | Purpose |
|---|---|
| `PLUTCHIK_API_KEY` | 64-char hex string for API authorization |
| `OPENROUTER_API_KEY` | Nemotron-3 comparative analysis |
| `HF_TOKEN` | Hugging Face Spaces updates |
| `DATABASE_URL` | Defaults to `sqlite:///plutchik_erc.db` |
| `PLUTCHIK_CDA_JSONL` | Path to CDA contrastive pairs JSONL |
| `PLUTCHIK_CDA_MIN_PAIRS` | Minimum verified pairs for CDA merge (default 200) |

### 5.4 Local Execution

```bash
# 1. Start the Inference Server
python3 inference_server.py

# 2. In a separate terminal, start the Dashboard
streamlit run app.py
```

### 5.5 Batch Processing

Dashboard supports multi-column CSVs:
- **Mandatory column**: `text`
- **Optional columns**: `speaker`, `topic`, `scenario`
- If optional columns are missing, sidebar defaults are used. Up to 2,000 rows per upload.

---

## 6. File Reference

| Module | Purpose |
|---|---|
| `app.py` | Streamlit dashboard & visualization |
| `inference_server.py` | FastAPI production API & lifecycle |
| `train_v2.py` | Training harness with wheel-distance weighting |
| `models/multitask_emotion_model.py` | Multi-task RoBERTa architecture & loss |
| `core/advanced_engine.py` | Neural ODEs, incongruity detection, reframing |
| `utils/constants.py` | Plutchik taxonomy, wheel-distance metric |
| `utils/preprocessing.py` | Dataset loading, context augmentation, IAA validation |
| `utils/trainer.py` | Training loop with AMP & gradient clipping |
| `utils/explainability_v2.py` | Captum Integrated Gradients |
| `utils/llm_inference.py` | Nemotron cross-validation client |
| `utils/pair_verifier.py` | Human-in-the-loop CDA verification |
| `utils/flywheel_trainer.py` | HITL correction fine-tuning |
| `database.py` | SQLAlchemy persistence layer |

---

**"Emotion is the DNA of conversation. Plutchik is the sequencer."**

(c) 2026 Plutchik ERC Project
