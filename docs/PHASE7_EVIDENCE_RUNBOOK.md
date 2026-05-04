# Phase 7 — Evidence runbook (Stage 2B / post-DAT)

Use this checklist after each training run that touches the DAT (GRL) shield.

## 1. Training (DAT + warmup)

From `plutchik_erc_dashboard/`:

```bash
python train_v2.py
```

Warmup is **epoch-based**: `warmup_epochs` full epochs with `alpha=0`, then GRL ramps over remaining steps (see `utils/trainer.py`). With `epochs=5` and `warmup_epochs=1`, the warmup is **20% of total optimizer steps** (one epoch of batches).

Optional CDA merge (human-verified contrastive JSONL):

```bash
export PLUTCHIK_CDA_JSONL=data/processed/ERC/contrastive_pairs.jsonl
export PLUTCHIK_CDA_MIN_PAIRS=200
python train_v2.py
```

## 2. Stage 2B done-criteria (log manually)

After convergence, record:

- Discriminator accuracy on the reversed scenario head (target **≤ 0.55** when shortcut is suppressed).
- Sarcasm F1 gap across scenarios (**< 0.15** between best and worst scenario).
- Emotion macro-F1 regression vs pre-DAT checkpoint (**< 0.04** absolute).

## 3. Bias re-audit (post-DAT)

From repo root (requires project venv with torch, sklearn, pandas):

```bash
python scripts/bias_audit.py \
  --csv data/processed/ERC/plutchik_v2_production.csv \
  --checkpoint plutchik_erc_dashboard/my_plutchik_model/best_model.pt --write-json
```

This writes `routing_decision.json` beside the checkpoint.

## 4. CDA gate count

```bash
python scripts/count_cda_verified.py data/processed/ERC/contrastive_pairs.jsonl
```

Stage 3 expects **≥ 200** `human_verified` rows before relying on the dissonance head in production.

## 5. Paste results

Append metrics and dates to [HARDENED_ARCHITECTURE_ROADMAP.md](HARDENED_ARCHITECTURE_ROADMAP.md) under **Phase 7 results log**.
