# Model Architecture Specification: Plutchik ERC v2.1

This document outlines the custom engineering behind the **Contextual Multi-Task RoBERTa** used for the Plutchik Emotion Recognition in Conversation benchmark.

## 1. Core Architecture
- **Backbone:** `roberta-base` (12-layer, 768-hidden, 12-heads, 125M parameters).
- **Rationale:** RoBERTa's robust pre-training on diverse linguistic patterns makes it superior to standard BERT for detecting subtle sarcasm and emotional subtext.
- **Input Representation:** 512 Max Tokens (Context-Aware).

## 2. Multi-Task Heads
The model processes a **768-dimensional vector** (from the `[CLS]` token) through three parallel specialized heads:

| Task | Type | Output | Description |
| :--- | :--- | :--- | :--- |
| **Emotion** | Classification | 32-way Softmax | Maps to the full Plutchik wheel. |
| **Sarcasm** | Classification | Binary (0/1) | Detects semantic dissonance between text and context. |
| **Intensity** | Regression | 1.0 - 3.0 | Estimates the "Ring" of the emotion (Mild → Intense). |

## 3. Engineering Features

### Contextual Memory (Sliding Window)
The model consumes a concatenated sequence of turns:  
`[Scenario] [Topic] [Turn T-2] [Turn T-1] [Current Turn]`  
This allows the self-attention mechanism to identify "Emotion Shifts" and cross-turn references.

### IAA-Weighted Loss
Training utilizes human annotation metadata to adjust the learning gradient:
- **High Agreement:** Penalizes the model heavily for errors (Strict Learning).
- **Low Agreement:** Reduces penalty to accommodate inherent emotional ambiguity (Soft Learning).

## 4. Explainability Framework
- **Engine:** Captum (PyTorch).
- **Method:** **Integrated Gradients (IG)**.
- **Output:** Token-level attribution mapping, identifying exactly which words triggered specific task predictions.

---
**Version:** 2.1 (Production)  
**Framework:** PyTorch / Transformers / Captum
