# CDA (Stage 3) verification gate

Training merges contrastive dissonance samples only when:

1. `PLUTCHIK_CDA_JSONL` points at a JSONL file on disk.
2. The file contains at least `PLUTCHIK_CDA_MIN_PAIRS` rows (default **200**) with `"human_verified": true`.

Produce verified rows with the interactive CLI:

```bash
cd plutchik_erc_dashboard
python utils/pair_verifier.py
```

Count progress anytime:

```bash
python scripts/count_cda_verified.py path/to/contrastive_pairs.jsonl
```

Templates live in `pair_templates.jsonl`; twins must satisfy `pair_verifier.VALIDATION_RULES` before human approval.
