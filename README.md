---
title: Plutchik Emotion Engine
emoji: 🎭
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 Plutchik Emotion Recognition Dashboard

A production-grade Emotion AI dashboard using the **Plutchik ERC Dataset v2.1** with a multi-task deep learning model and interactive explainability visualizations.

---

## 🎯 Features

### **Phase 1: Multi-Task Model**
- **32-class Emotion Classification** (Plutchik emotions)
- **Binary Sarcasm Detection** (sarcasm_flag prediction)
- **Intensity Regression** (0-1 scale per ring: mild, primary, intense, dyadic)
- **Weighted Multi-Task Loss** combining CrossEntropy + MSE
- **Inter-Annotator Agreement (IAA) Weighting** for robust training

### **Phase 2: Deep ERC Preprocessing**
- **Metadata Augmentation**: Prepends `[SCENARIO]` and `[TOPIC]` tags
- **Context Window**: Retrieves previous 2 dialogue turns for emotion shift detection
- **Token-level Processing**: RoBERTa tokenization with attention masks

### **Phase 3: Explainability Pipeline**
- **Token Visualization**: Shows active tokens with IDs
- **Embedding Heatmap**: 768D → sampled 30D for visualization
- **Cosine Similarity Analysis**: Compares input to learned emotion centroids
- **PCA Reduction**: 2D/3D projection of embedding space

### **Phase 4: Streamlit UI**
- **Radar Chart** (Spider plot of 8 primary Plutchik sectors)
- **Intensity Gauge** (visual indicator: mild → intense)
- **Sarcasm Detector** (percentage bar)
- **Top-5 Emotions** bar chart
- **Interactive Explainability Tabs**:
  - 🔍 Tokenization breakdown
  - 📊 Embedding heatmap
  - 🎯 Cosine similarity scores

---

## 📁 Project Structure

```
plutchik_erc_dashboard/
├── models/
│   └── multitask_emotion_model.py       # PluTchikMultiTaskModel + MultiTaskLoss
├── utils/
│   ├── preprocessing.py                  # ERCPreprocessor + PlutchikERCDataset
│   ├── explainability.py                 # ExplainabilityEngine
│   └── trainer.py                        # PluTchikTrainer
├── my_plutchik_model/                    # Model checkpoints & centroids
│   ├── best_model.pt                     # Saved checkpoint
│   └── emotion_centroids.pkl             # Emotion centroids for similarity
├── app.py                                # Streamlit dashboard
├── train.py                              # Training & inference script
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 🚀 Quick Start

### 1. **Setup**

```bash
cd plutchik_erc_dashboard
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. **Train Model**

```bash
python train.py --train
```

**Output:**
- Trains for 3 epochs (configurable)
- Saves best checkpoint → `my_plutchik_model/best_model.pt`
- Saves emotion centroids → `my_plutchik_model/emotion_centroids.pkl`

### 3. **Run Dashboard**

```bash
streamlit run app.py
```

**Access:** `http://localhost:8501`

### 4. **Inference Only**

```bash
python train.py --infer
```

---

## 📊 Model Architecture

```
Input Text
    ↓
[Augment with SCENARIO + TOPIC metadata]
    ↓
RoBERTa Tokenizer (256 tokens max)
    ↓
RoBERTa-base backbone (768D hidden)
    ↓
Shared Dense Layer (LayerNorm + Dropout)
    ├─→ Emotion Head → 32-way classification
    ├─→ Sarcasm Head → Binary classification
    └─→ Intensity Head → MSE regression [0,1]
    ↓
Multi-Task Loss (weighted combination)
```

### **Loss Function**
```
L_total = 1.0 * L_emotion + 0.5 * L_sarcasm + 0.3 * L_intensity
        × IAA_weight (inter-annotator agreement)
```

---

## 🎭 Plutchik Emotion Model

The dashboard recognizes **32 emotions** organized by:

### **Primary (8)**
`joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`

### **Intense (8)**
`ecstasy`, `admiration`, `terror`, `amazement`, `grief`, `loathing`, `rage`, `vigilance`

### **Mild (8)**
`serenity`, `acceptance`, `apprehension`, `distraction`, `pensiveness`, `boredom`, `annoyance`, `interest`

### **Dyadic (8)**
`optimism`, `love`, `submission`, `awe`, `disapproval`, `remorse`, `contempt`, `aggressiveness`

---

## 🔬 Explainability Features

### **Tokenization Tab**
- Shows all tokenized words with their token IDs
- Removes padding tokens automatically

### **Embeddings Tab**
- Heatmap of token embeddings (768D → 30 sampled dimensions)
- Highlights which regions of the embedding space are active

### **Cosine Similarity Tab**
- Computes similarity between input and each emotion's learned centroid
- **Why it works**: 
  - During training, embeddings of "grief" utterances cluster together
  - Input embedding is compared to centroid of each emotion
  - Higher similarity = stronger prediction signal

---

## 💾 Model Persistence

### **Checkpoint Format** (`best_model.pt`)
```python
{
    "epoch": 3,
    "model_state_dict": {...},
    "metrics": {...},
    "training_history": {...}
}
```

### **Emotion Centroids** (`emotion_centroids.pkl`)
```python
{
    0: np.ndarray([768]),   # joy centroid
    1: np.ndarray([768]),   # trust centroid
    ...
    31: np.ndarray([768])   # aggressiveness centroid
}
```

---

## 🔧 Configuration

### **Training Parameters** (in `train.py`)
```python
trainer.fit(
    train_loader,
    val_loader,
    epochs=3,              # Number of training epochs
    learning_rate=2e-5     # AdamW learning rate
)
```

### **Multi-Task Weights** (in `train.py`)
```python
loss_fn = MultiTaskLoss(
    emotion_weight=1.0,    # Primary task
    sarcasm_weight=0.5,    # Auxiliary task
    intensity_weight=0.3   # Auxiliary task
)
```

### **Preprocessing** (in `preprocessing.py`)
```python
# Context window size
window_size=2  # Previous 2 turns

# Max sequence length
max_length=256  # Tokens

# Intensity mapping
ring_to_intensity = {
    "intense": 1.0,
    "primary": 0.5,
    "mild": 0.25,
    "dyadic": 0.6
}
```

---

## 📈 Expected Performance

| Metric | Accuracy |
|--------|----------|
| Emotion Classification | ~75-85% (32-way) |
| Sarcasm Detection | ~80-90% (binary) |
| Intensity MSE | ~0.05-0.10 |

*Note: Depends on dataset size and training epochs. This demo trains on limited data.*

---

## 🎨 UI Components

### **Metrics Row**
- 🎭 **Emotion**: Predicted emotion + confidence
- 🤔 **Sarcasm**: Percentage (>50% = sarcasm detected)
- 📊 **Intensity Ring**: Mild/Primary/Intense/Dyadic
- 💪 **Intensity Level**: Numeric 0-1 score

### **Radar Chart**
- 8-sector radial plot of primary emotions
- Shows probability distribution across 8 base emotions

### **Intensity Gauge**
- Colored gauge matching emotion ring
- Ranges: Mild (pale) → Primary (orange) → Intense (red)

### **Top 5 Emotions**
- Horizontal bar chart
- Predicted emotion highlighted in red

### **Explainability Tabs**
- Token viewer
- Embedding heatmap
- Similarity heatmap

---

## 🔄 Workflow Example

1. **User Input:**
   ```
   Scenario: workplace
   Topic: termination
   Text: "I have been with this organisation for six years without a single negative performance review."
   ```

2. **Preprocessing:**
   ```
   [SCENARIO] workplace [/SCENARIO] [TOPIC] termination [/TOPIC] I have been...
   + Context from previous 2 turns
   ```

3. **Tokenization:**
   ```
   [CLS] [SCENARIO] workplace [/SCENARIO] ... [SEP]
   Token IDs: [101, 50264, 35867, ...]
   ```

4. **Forward Pass:**
   - RoBERTa encodes to 768D vectors
   - Shared layer processes CLS token
   - 3 heads predict simultaneously

5. **Output:**
   ```
   Emotion: rage (78% confidence)
   Sarcasm: No (12%)
   Intensity: 0.8 (primary ring)
   ```

6. **Explainability:**
   - Shows which tokens were important
   - Displays embedding similarity to "rage" centroid (0.82)

---

## 🛠️ Advanced Usage

### **Custom Training with CSV Data**

```python
from preprocessing import build_dataset_from_dialogues
from models import PluTchikMultiTaskModel, MultiTaskLoss
from utils.trainer import PluTchikTrainer

# Load your CSV with columns: scenario, topic, utterances, emotion, sarcasm, intensity, iaa
dialogues = load_dialogues_from_csv("data.csv")
dataset = build_dataset_from_dialogues(dialogues, PLUTCHIK)

# Train
trainer.fit(train_loader, val_loader, epochs=10)
```

### **Extract Embeddings for Downstream Tasks**

```python
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    cls_embedding = outputs["cls_embedding"]  # [batch, 768]
    
# Use cls_embedding for clustering, visualization, similarity, etc.
```

### **Visualize Emotion Clusters**

```python
from sklearn.manifold import TSNE

# Gather all CLS embeddings from validation set
all_embeddings = np.vstack([...])  # [n_samples, 768]

# TSNE reduction
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(all_embeddings)

# Plot with emotion labels for analysis
```

---

## ⚙️ System Requirements

- **Python:** 3.9+
- **GPU:** Optional (CPU supported, slower)
- **RAM:** 8GB+ recommended
- **Disk:** 2GB+ for model + dependencies

---

## 📝 Citation

```bibtex
@dataset{plutchik_erc_v2,
  title={Plutchik Emotion Recognition in Conversation Dataset v2.1},
  year={2024},
  note={32 emotions, 18 metadata columns}
}
```

---

## 🤝 Support

For issues or improvements:
1. Check the error message
2. Verify all dependencies are installed
3. Ensure `my_plutchik_model/` directory exists
4. Try reinstalling: `pip install -r requirements.txt --force-reinstall`

---

## 📚 References

- **Plutchik Emotion Model**: [Theory](https://en.wikipedia.org/wiki/Plutchik's_wheel_of_emotions)
- **RoBERTa**: [Paper](https://arxiv.org/abs/1907.11692)
- **Multi-Task Learning**: [Survey](https://arxiv.org/abs/1506.00863)
- **Emotion Recognition in Conversation**: [Review](https://arxiv.org/abs/2105.02727)

---

**Built with ❤️ for emotion AI research.**
