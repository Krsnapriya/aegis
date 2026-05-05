# 🎭 The Ultimate Plutchik Emotion Engine: Technical Deep Dive & Production Manual

[![Version: 2.5.0-Hardened](https://img.shields.io/badge/Version-2.5.0--Hardened-brightgreen.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)
[![Architecture: Multi-Task Transformer](https://img.shields.io/badge/Architecture-Multi--Task--Transformer-blue.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)
[![Taxonomy: Plutchik 32](https://img.shields.io/badge/Taxonomy-Plutchik--32-orange.svg?style=for-the-badge)](https://github.com/Krsnapriya/aegis)

---

## 📖 1. Theoretical Foundation: The Psychophysiology of Plutchik

The core of this engine is grounded in **Robert Plutchik's Psychoevolutionary Theory of Emotion**. Unlike models based on Ekman's "Big Six" (which are discrete), Plutchik's model is **structured and hierarchical**. 

### 1.1 The Wheel of Emotion
Plutchik's wheel consists of 8 primary bipolar emotions: **Joy vs. Sadness**, **Anger vs. Fear**, **Trust vs. Disgust**, and **Surprise vs. Anticipation**. 
*   **Dimensionality**: Each emotion exists on a vertical axis of **Intensity**. For example: *Annoyance → Anger → Rage*. 
*   **Dyads**: Plutchik proposed that complex emotions are mixtures of these primary ones. *Love = Joy + Trust*; *Optimism = Anticipation + Joy*.
*   **The 32-Class Taxonomy**: This engine implements a refined 32-class version of this taxonomy, capturing the "Secondary" and "Tertiary" rings of the wheel to provide unprecedented granularity in conversational analysis.

### 1.2 Why ERC (Emotion Recognition in Conversation)?
Standard sentiment analysis is "stateless." It looks at a single sentence in isolation. **ERC** is "stateful." It recognizes that an utterance like "I'm fine" means something entirely different after a heated argument than after a positive resolution. 
*   **Contextual Shifting**: Emotions in conversation are dynamic. The engine tracks the "Emotional Trajectory" across multiple turns.
*   **Persona Bias**: An "Agent" in a support scenario has a different emotional baseline than a "Customer."

---

## 🏗️ 2. Architectural Blueprint: The Hardened Production Suite

The system is designed as a **decoupled, high-performance microservice architecture**.

### 2.1 Component Breakdown

#### A. The Neural Core (PyTorch + RoBERTa)
At the center is a **Multi-Task RoBERTa-Base** model. We chose RoBERTa over BERT due to its improved pre-training on larger datasets and removal of the Next Sentence Prediction (NSP) task, which we found less relevant for short-turn conversational dynamics.
*   **Shared Encoder**: A 12-layer, 768-hidden Transformer block.
*   **Task-Specific Heads**: Four non-linear MLP heads that branch from the CLS token representation.

#### B. The Inference Gateway (FastAPI)
`inference_server.py` is the production entry point. It manages:
*   **Model Lifecycle**: Loading, versioning, and hot-reloading.
*   **Auth & Security**: X-API-Key validation for multi-tenant isolation.
*   **Resource Management**: Intelligent switching between CUDA/MPS/CPU.

#### C. The Intelligence Hub (Streamlit)
`app.py` provides the "Command Center." It is optimized for **Human-in-the-Loop (HITL)** workflows, allowing researchers to audit model reasoning in real-time using Captum-based heatmaps.

---

## 🧠 3. Algorithmic Deep Dive: The Mathematics of Emotion

### 3.1 Multi-Task Learning (MTL) & Weighted Loss
The engine solves four tasks simultaneously. The total loss $L$ is a weighted sum:

$$L_{total} = \lambda_1 L_{CE\_emo} + \lambda_2 L_{CE\_sarc} + \lambda_3 L_{MSE\_int} + \lambda_4 L_{CE\_adv}$$

1.  **Emotion ($L_{CE\_emo}$)**: 32-way Cross-Entropy loss.
2.  **Sarcasm ($L_{CE\_sarc}$)**: Binary Cross-Entropy. Crucial for detecting dissonance.
3.  **Intensity ($L_{MSE\_int}$)**: Mean Squared Error regression. Captures the "distance from the wheel's center."
4.  **Adversarial Domain Lock ($\lambda_4 L_{CE\_adv}$)**: Uses a Gradient Reversal Layer (GRL). During backpropagation, the gradients from this head are multiplied by $-\lambda$, forcing the encoder to learn features that are *not* indicative of the scenario (e.g., removing "Workplace" bias from "Anger" detection).

### 3.2 Explainability: Integrated Gradients (IG)
We implement the **Integrated Gradients** algorithm (Sundararajan et al.). Unlike simple Gradient * Input methods, IG satisfies the "Axiom of Completeness."
*   **Formula**: $IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$
*   **Implementation**: We approximate the integral using a Gauss-Legendre quadrature (typically 50-100 steps). This allows the dashboard to highlight tokens with **Positive Attribution** (supporting the prediction) and **Negative Attribution** (contradicting the prediction).

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

#### `inference_server.py`
The "Brain." Engineered for stability.
*   **Async Write Queue**: Implements a `queue.Queue` with a background worker thread (`_db_worker`). This ensures that even if SQLite is locked under load, predictions are still returned to the user instantly, with DB persistence happening in the background.
*   **Reload API**: A POST `/reload` endpoint that re-triggers the model's weight-loading sequence.

#### `train_v2.py`
The "Forge." The primary training script.
*   **Hardening Logic**: Implements "Seed Injection" where specific high-confidence manual examples are prioritized in the training batch to prevent "Drift."

---

### 4.2 Models & Architecture (`models/`)

#### `multitask_emotion_model.py`
*   **`PluTchikMultiTaskModel`**: The PyTorch implementation. Includes the `GradReverse` function for adversarial training and the dual-encoder pathway for dissonance detection.
*   **`MultiTaskLoss`**: The custom loss class that handles multi-task weighting and Inter-Annotator Agreement (IAA) weighting.

#### `db_models.py`
Defines the `SignalAudit` and `SarcasmEvent` schemas. Used by SQLAlchemy to ensure type-safe persistence.

---

### 4.3 Utilities & Pipelines (`utils/`)

#### `preprocessing.py`
The "Filter."
*   **`ERCPreprocessor`**: Handles the critical `[SCENARIO]` and `[CONTEXT]` token injection. It manages a sliding window of dialogue history, ensuring that every prediction "sees" the previous $N$ turns.
*   **Scenario Mapping**: Maps 14+ scenarios (Romance, Workplace, etc.) into binary domains for the Adversarial head.

#### `explainability_v2.py`
*   **`CaptumExplainer`**: A specialized wrapper for the Captum library. It handles the mapping of sub-word BPE tokens back to original user-visible words, aggregating attributions to make them readable.

#### `llm_inference.py`
*   **`NemotronClient`**: A resilient client for OpenRouter. It handles LLM fallbacks, retry logic, and parses the "Reasoning" blocks from the 120B parameter model.

#### `trainer.py`
A generalized training loop with support for **F1-Macro scoring**, checkpointing, and early stopping.

---

## 🚀 5. Operational Guide: Deployment & Scaling

### 5.1 Environment Variables
*   `PLUTCHIK_API_KEY`: 64-char hex string for API authorization.
*   `OPENROUTER_API_KEY`: Required for Nemotron-3 comparative analysis.
*   `DATABASE_URL`: Defaults to `sqlite:///plutchik_erc.db`. Can be pointed to a PostgreSQL instance for production.

### 5.2 Performance Optimization
*   **MPS Acceleration**: Optimized for Mac Silicon (M1/M2/M3). Training uses `torch.device("mps")`.
*   **Batching**: The `/predict/batch` endpoint uses vectorization to process up to 200 utterances in a single forward pass, significantly reducing overhead compared to sequential calls.

---

## 🔮 6. The Roadmap: Future Iterations

1.  **Cross-Modal Expansion**: Incorporating audio-prosody features (pitch, tone) alongside text for a multi-modal Plutchik Engine.
2.  **Federated Learning**: Allowing edge deployment where the model learns from local user "Corrections" without transmitting raw text to a central server.
3.  **Real-Time VAD (Voice Activity Detection)**: Integration with live telephony streams.

---

## 📜 Summary Table of Components

| Module | Technical Specialization | Purpose |
| :--- | :--- | :--- |
| **`app.py`** | Streamlit / CSS | Frontend Dashboard & Visualization |
| **`inference_server.py`** | FastAPI / Uvicorn | Production API & Lifecycle Management |
| **`train_v2.py`** | PyTorch / Training | Model Fine-tuning & Weight Convergence |
| **`advanced_engine.py`** | ODEs / Dynamics | Trajectory Forecasting & State Dynamics |
| **`preprocessing.py`** | NLP / Tokenization | Context Windowing & Metadata Injection |
| **`multitask_model.py`** | MTL / Transformers | Core Neural Network (RoBERTa) |
| **`llm_inference.py`** | API / OpenRouter | LLM Comparision & Teacher Logic |

---

### **"Emotion is the DNA of conversation. Plutchik is the sequencer."**
