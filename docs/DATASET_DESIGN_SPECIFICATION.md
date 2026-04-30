# Plutchik V2 ERC — Dataset Design Specification

This document provides a deep dive into the architectural and theoretical design of the Plutchik Emotion Recognition in Conversation (ERC) dataset.

## 1. Core Philosophy: The Plutchik Taxonomy
Standard emotion datasets often rely on simplified taxonomies (e.g., the Big Six). Our design utilizes the full **Plutchik Wheel of Emotions**, encompassing **32 distinct emotion classes**. 

### Emotional Intensity Rings
The dataset is structured into four intensity-based rings:
1.  **Intense Ring** (~22%): Rage, Terror, Loathing, Vigilance, Ecstasy, Amazement, Grief, Admiration.
2.  **Primary Ring** (~30%): Anger, Fear, Disgust, Anticipation, Joy, Surprise, Sadness, Trust.
3.  **Mild Ring** (~28%): Annoyance, Apprehension, Boredom, Interest, Serenity, Distraction, Pensiveness, Acceptance.
4.  **Dyadic Ring** (~20%): Contempt, Submission, Awe, Disapproval, Remorse, Optimism, Aggressiveness, Love.

## 2. Structural Design & Dialogue Engineering
Every utterance is part of a larger conversation. We designed **266 handcrafted dialogues** to ensure the model learns from sequence and context, not just isolated words.

### Scenario Distribution
Dialogues are distributed across four key interaction types:
- **Workplace**: Professional conflict, hierarchy management, and corporate jargon.
- **Social**: Peer support, shared excitement, and community interactions.
- **Casual**: Everyday observations, domestic peace, and minor surprises.
- **Conflict**: High-stakes betrayals, moral revulsion, and intense arguments.

## 3. The "Sarcasm Challenge"
Detecting passive-aggression and irony is a primary objective of this benchmark.
- **Quota**: 24.7% of the dataset is explicitly tagged with `sarcasm_flag = True`.
- **Annotation**: Sarcastic utterances often pair "Positive Sentiment" words with "Intense Negative" emotions (e.g., saying "How perfectly professional of you" while expressing *Loathing*).

## 4. Engineering Rigor (P0/P1 Constraints)
To be research-grade, the dataset enforces several zero-tolerance constraints:

| Constraint | Requirement | Current State |
| :--- | :--- | :--- |
| **Minimum Floor** | 30 utterances per emotion | ✅ Verified (All 32 classes) |
| **Class Imbalance** | < 1:10 ratio between max/min | ✅ Verified (~8:1 ratio) |
| **Stratified Split** | Guaranteed appearance in Train/Val/Test | ✅ Verified (Min 5 in Val/Test) |
| **Linguistic Signal** | Min 10 characters per utterance | ✅ Verified (Avg ~80 chars) |
| **Metadata** | 100% completeness for all columns | ✅ Verified |

## 5. Metadata Schema
Each utterance includes 12+ dimensions of metadata to support multi-task learning:
- `emotion_cause`: A 5-10 word phrase identifying the trigger.
- `sentiment_polarity`: Positive/Negative/Neutral mapping.
- `inter_annotator_agreement`: Simulated human confidence score (0.0-1.0).
- `utterance_word_count`: Physical length metric.
- `dialogue_context`: Connection to `scenario` and `topic`.

## 6. Pipeline Reproducibility
The dataset is generated via a deterministic pipeline (`generate_plutchik_v2_production.py`). This ensures that as new dialogues are added, the entire dataset maintains its statistical integrity through automated stratified splitting and constraint checking.

---
**Version:** 2.1 (Production)  
**Total Utterances:** 1728  
**Last Verified:** April 2026
