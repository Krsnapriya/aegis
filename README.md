---
title: Plutchik Emotion Engine
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# 🎭 The Ultimate Plutchik Emotion Engine: Technical Deep Dive & Production Manual

[![Version: 2.5.0-Hardened](https://img.shields.io/badge/Version-2.5.0--Hardened-brightgreen.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)
[![Architecture: Multi-Task Transformer](https://img.shields.io/badge/Architecture-Multi--Task--Transformer-blue.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)
[![Taxonomy: Plutchik 32](https://img.shields.io/badge/Taxonomy-Plutchik--32-orange.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)

---

## 📖 1. Theoretical Foundation: The Psychophysiology of Plutchik

The core of this engine is grounded in **Robert Plutchik's Psychoevolutionary Theory of Emotion**. Unlike models based on Ekman's "Big Six" (which are discrete), Plutchik's model is **structured and hierarchical**. 

### 1.1 The Structural Model (The Wheel)
Plutchik's 1980 model proposed eight primary bipolar emotions, paired as opposites:
*   **Joy vs. Sadness**
*   **Anger vs. Fear**
*   **Trust vs. Disgust**
*   **Surprise vs. Anticipation**

#### The Three Dimensions:
1.  **Intensity**: Emotions vary in strength. *Annoyance* escalates to *Anger*, which peaks at *Rage*. The engine models this as a continuous regression value [0,1].
2.  **Similarity**: Emotions located closer on the wheel (e.g., Joy and Trust) are more likely to co-occur.
3.  **Polarity**: Opposite emotions (e.g., Joy and Sadness) rarely manifest simultaneously without creating deep psychological dissonance.

### 1.2 The 32-Class Taxonomy
This engine implements a refined 32-class version of this taxonomy, capturing:
*   **Primary Emotions**: The 8 basic states.
*   **Secondary Dyads**: Combinations like *Love* (Joy + Trust) or *Optimism* (Anticipation + Joy).
*   **Tertiary Rings**: Subtle variants like *Apprehension* (low-intensity Fear) or *Pensiveness* (low-intensity Sadness).

### 1.3 Why ERC (Emotion Recognition in Conversation)?
Traditional sentiment analysis is "stateless." It looks at a single sentence like "I'm fine" and might classify it as Neutral. **ERC** is "stateful." It recognizes that "I'm fine" spoken after a long silence or a biting remark is likely a signal of **Disgust** or **Anger (passive-aggressive)**. 
*   **Contextual Shifting**: Emotions in conversation are dynamic. The engine tracks the "Emotional Trajectory" across multiple turns.
*   **Persona Bias**: An "Agent" in a support scenario has a different emotional baseline than a "Customer."

---

## 🏗️ 2. Architectural Blueprint: The Hardened Production Suite

The system is designed as a **decoupled, high-performance microservice architecture**.

### 2.1 Component Breakdown

#### A. The Neural Core (PyTorch + RoBERTa)
At the center is a **Multi-Task RoBERTa-Base** model. We chose RoBERTa over BERT due to its improved pre-training on larger datasets and removal of the Next Sentence Prediction (NSP) task, which we found less relevant for short-turn conversational dynamics.
*   **Shared Encoder**: A 12-layer, 768-hidden Transformer block.
*   **Task-Specific Heads**: Four non-linear MLP heads branch from the CLS token representation.

#### B. The Inference Gateway (FastAPI)
`inference_server.py` is the production entry point. It manages:
*   **Model Lifecycle**: Loading, versioning, and hot-reloading.
*   **Auth & Security**: X-API-Key validation for multi-tenant isolation.
*   **Resource Management**: Intelligent switching between CUDA/MPS/CPU.
*   **Async Write Queue**: A multi-threaded worker that handles database persistence without blocking prediction responses.

#### C. The Intelligence Hub (Streamlit)
`app.py` provides the "Command Center." It is optimized for **Human-in-the-Loop (HITL)** workflows, allowing researchers to audit model reasoning in real-time using Captum-based heatmaps.

---

## 🧠 3. Algorithmic Deep Dive: The Mathematics of Emotion

### 3.1 Multi-Task Learning (MTL) & Weighted Loss
The engine solves four tasks simultaneously. The total loss $L$ is a weighted sum:

$$L_{total} = \lambda_1 L_{CE\_emo} + \lambda_2 L_{CE\_sarc} + \lambda_3 L_{MSE\_int} + \lambda_4 L_{CE\_adv}$$

1.  **Emotion ($L_{CE\_emo}$)**: 32-way Cross-Entropy loss. This is the primary driver of emotional categorization.
2.  **Sarcasm ($L_{CE\_sarc}$)**: Binary Cross-Entropy. Crucial for detecting dissonance. When sarcasm is high, the model "distrusts" the surface emotion.
3.  **Intensity ($L_{MSE\_int}$)**: Mean Squared Error regression. Captures the "distance from the wheel's center."
4.  **Adversarial Domain Lock ($\lambda_4 L_{CE\_adv}$)**: Uses a Gradient Reversal Layer (GRL). During backpropagation, the gradients from this head are multiplied by $-\lambda$, forcing the encoder to learn features that are *not* indicative of the scenario (e.g., removing "Workplace" bias from "Anger" detection).

### 3.2 Explainability: Integrated Gradients (IG)
We implement the **Integrated Gradients** algorithm (Sundararajan et al.). Unlike simple Gradient * Input methods, IG satisfies the "Axiom of Completeness."
*   **Formula**: $IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$
*   **Implementation**: We approximate the integral using a Gauss-Legendre quadrature (typically 50 steps). This allows the dashboard to highlight tokens with **Positive Attribution** (supporting the prediction) and **Negative Attribution** (contradicting the prediction).

### 3.3 Dynamic Intelligence: Emotion ODEs
Located in `core/advanced_engine.py`, this module treats emotional states as a **Continuous Dynamical System**.
*   **The Problem**: Conversational turns happen at discrete intervals ($t=1, 2, 3$), but human emotion is continuous.
*   **The Solution**: We model the "Latent Emotion Vector" $h$ as following an Ordinary Differential Equation:
    $$\frac{dh}{dt} = f(h(t), t, \theta)$$
    where $f$ is a neural network (`EmotionODEFunc`). This allows us to "interpolate" emotional states between turns and "extrapolate" future emotional escalation.

---

## 📂 4. The Technical Directory: File-by-File Analysis

### 4.1 Core Application Layer

#### `app.py`
The "Face" of the project. It uses a custom **Glassmorphism CSS** layer to provide a premium interface.
*   **Key Logic**: Manages session state, handles the "Dynamic Intelligence" loop, and orchestrates the comparison between Local RoBERTa and LLM (Nemotron) models.
*   **Batch Upload**: Utilizes `st.file_uploader` and Pandas to score thousands of rows asynchronously.
*   **Visualizations**: Uses Plotly for Radar charts (emotional spectrum), Area charts (intensity trajectories), and Heatmaps (attributions).

#### `inference_server.py`
The "Brain." Engineered for high availability.
*   **Hot-Reload API**: A POST `/reload` endpoint that re-triggers the model's weight-loading sequence.
*   **Safety Rails**: Implements Pydantic validation on all inputs, sanitizing text and enforcing length limits (5000 chars) to prevent OOM.
*   **Persistence Strategy**: Uses SQLAlchemy with a single-writer background thread to survive SQLite lock storms under high concurrency.

#### `train_v2.py`
The "Forge." The primary training script.
*   **Hardening Logic**: Implements "Seed Injection" where specific high-confidence manual examples are prioritized in the training batch to prevent "Drift."
*   **Acceleration**: Automatically detects and utilizes `mps` (Mac Silicon) or `cuda` (NVIDIA) backends.

---

### 4.2 Models & Architecture (`models/`)

#### `multitask_emotion_model.py`
*   **`PluTchikMultiTaskModel`**: The core PyTorch implementation.
*   **`GradReverse`**: A custom `torch.autograd.Function` that implements the gradient reversal for adversarial training.
*   **Dissonance Head**: A dual-encoder architecture that accepts both a "Context" sequence and a "Current" sequence to calculate semantic mismatch.

#### `db_models.py`
Defines the `SignalAudit` schema. Every prediction is logged with its full metadata (timestamp, speaker, scenario, topic, confidence, intensity) for later auditing.

---

### 4.3 Utilities & Pipelines (`utils/`)

#### `preprocessing.py`
The "Filter."
*   **`ERCPreprocessor`**: Handles the critical `[SCENARIO]` and `[CONTEXT]` token injection. It manages a sliding window of dialogue history, ensuring that every prediction "sees" the previous $N$ turns.
*   **Metadata Augmentation**: Formats inputs as `[CONTEXT] turn1 | turn2 [/CONTEXT] [CURRENT] [SCENARIO] workplace [/SCENARIO] text [/CURRENT]`.

#### `explainability_v2.py`
*   **`CaptumExplainer`**: A specialized wrapper for the Captum library. It handles the mapping of sub-word BPE tokens back to original user-visible words, aggregating attributions to make them readable.

#### `llm_inference.py`
*   **`NemotronClient`**: A resilient client for OpenRouter. It handles LLM fallbacks (keyword-based), retry logic, and parses the "Reasoning" blocks from the 120B parameter model.

#### `trainer.py`
A robust `Trainer` class implementing:
*   **Gradient Clipping**: Prevents exploding gradients in the Transformer block.
*   **Mixed-Precision (AMP)**: Uses `torch.amp` to speed up training on supported GPUs.
*   **Validation Metrics**: Calculates F1-Macro and Accuracy per task.

---

## 🚀 5. Operational Guide: Deployment & Scaling

### 5.1 Environment Variables
*   `PLUTCHIK_API_KEY`: 64-char hex string for API authorization.
*   `OPENROUTER_API_KEY`: Required for Nemotron-3 comparative analysis.
*   `HF_TOKEN`: Required for pushing updates to Hugging Face Spaces.
*   `DATABASE_URL`: Defaults to `sqlite:///plutchik_erc.db`.

### 5.2 Local Execution
```bash
# 1. Start the Inference Server
python3 inference_server.py

# 2. In a separate terminal, start the Dashboard
streamlit run app.py
```

### 5.3 Batch Processing Nuances
The Dashboard now supports multi-column CSVs.
*   **Mandatory Column**: `text`
*   **Optional Columns**: `speaker`, `topic`, `scenario`
*   **Behavior**: If optional columns are missing, the system uses the sidebar defaults. If present, it uses the specific metadata for each individual row.

---

---

## 📈 6. Hardening & Performance

### 6.1 Accuracy Benchmarks (v2.5)
*   **Training Accuracy**: 80.2%
*   **Validation Accuracy**: 54.1%
*   **F1-Macro (Emotion)**: 0.45
*   **Inference Latency (Standard)**: ~250ms
*   **Inference Latency (Fast Attribution)**: ~800ms (CPU) / ~150ms (GPU)
*   **Inference Latency (Full Captum)**: ~15s (CPU) / ~2s (GPU)

### 6.2 Convergence Strategy
The current model was trained for **5 epochs** with a **Cosine Annealing** learning rate scheduler, starting at `2e-5`. The first 2 epochs were "Warmup" phases where the Adversarial Discriminator was slowly phased in to prevent it from destabilizing the early emotional learning.

### 6.3 Recent Optimizations (v2.5.1)
*   **Riemann Approximation (Fast IG)**: Token attributions now utilize a 5-step Riemann approximation by default for the dashboard. This provides a 10x speedup in explainability results without sacrificing the core directional attribution markers.
*   **Robust Path Resolution**: Implemented `sys.path` injection and `.env`-driven `PYTHONPATH` configuration to ensure the engine remains stable across variable execution environments and IDE configurations.
*   **Dependency Hardening**: Integrated `torchdiffeq` directly into the production environment to support continuous-time emotional forecasting (Neural ODEs).

---

## 📜 Technical Purpose Summary Table

| Module | Technical Specialization | Purpose |
| :--- | :--- | :--- |
| **`app.py`** | Streamlit / Plotly | User Interface & Real-time Visualization |
| **`inference_server.py`** | FastAPI / Async | Production API & Lifecycle Management |
| **`train_v2.py`** | PyTorch / AdamW | Model Fine-tuning & Checkpointing |
| **`advanced_engine.py`** | ODEs / Dynamics | Continuous State Forecasting |
| **`preprocessing.py`** | NLP / Tokenization | Context Augmentation & Metadata Injection |
| **`multitask_model.py`** | MTL / GRL | Core Neural Architecture |
| **`llm_inference.py`** | API / OpenRouter | LLM Cross-Validation & Teacher Logic |
| **`trainer.py`** | AMP / Sklearn | Cross-Platform Training Pipeline |

---

### **"Emotion is the DNA of conversation. Plutchik is the sequencer."**
© 2026 Plutchik ERC Project | Hardened for Production.
