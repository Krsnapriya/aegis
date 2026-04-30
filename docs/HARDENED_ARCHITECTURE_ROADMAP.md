# Hardened Architecture Roadmap: Plutchik ERC v2.1

This roadmap defines the gated execution plan for the next stages of the Plutchik project, focusing on adversarial robustness, contrastive learning, and real-world deployment.

## Stage 2: Adversarial Hardening (The "Antigravity" Shield)
**Goal:** Eliminate scenario-bias in sarcasm detection.

### 2B: Gradient Reversal Layer (GRL) & Lambda Warmup
- **Invariant:** The GRL lambda must have a warmup fraction (minimum 20%).
- **Risk:** Without warmup, the encoder may collapse before learning meaningful emotion features.
- **Done-Criteria:** Emotion F1 must remain stable (±2%) as the GRL head begins penalizing scenario-recognition.

## Stage 3: Contrastive Dissonance Augmentation (CDA)
**Goal:** Force the model to learn linguistic dissonance through twin-context pairs.

### Verification Gate (Manual Audit)
- **Requirement:** 200 verified pairs (Literal vs Sarcastic).
- **Process:** Human-in-the-loop review (~8 hours).
- **Warning:** Do NOT use a model-based verifier. The ground truth must be human-verified.

## Stage 4: Real-World Adaptation (Reddit Adapter)
**Goal:** Transition from synthetic benchmark to in-the-wild inference.

### DOM Target Verification
- **Target:** Reddit SPA.
- **Selector Check:** Confirm `[data-testid="comment"]` stability in the target browser environment.
- **Done-Criteria:** Successful extraction of a 3-turn context thread into the Antigravity JSON schema.

---
**Status:** Approved Roadmap  
**Version:** 2.1.H (Hardened)
