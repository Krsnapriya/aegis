"""
Plutchik ERC v2.1 — Canonical Constants
Single source of truth for the Plutchik emotion taxonomy.
All modules import from here. Do not duplicate this dictionary elsewhere.
"""
import torch

# ============== PLUTCHIK WHEEL OF EMOTIONS (32 Classes) ==============
PLUTCHIK = {
    # Primary Emotions (Ring 2)
    "joy": {"ring": "primary", "sector": "joy", "color": "F4D03F"},
    "trust": {"ring": "primary", "sector": "trust", "color": "27AE60"},
    "fear": {"ring": "primary", "sector": "fear", "color": "196F3D"},
    "surprise": {"ring": "primary", "sector": "surprise", "color": "2E86C1"},
    "sadness": {"ring": "primary", "sector": "sadness", "color": "2980B9"},
    "disgust": {"ring": "primary", "sector": "disgust", "color": "AF7AC5"},
    "anger": {"ring": "primary", "sector": "anger", "color": "E74C3C"},
    "anticipation": {"ring": "primary", "sector": "anticipation", "color": "E67E22"},
    # Intense Emotions (Ring 3)
    "ecstasy": {"ring": "intense", "sector": "joy", "color": "F1C40F"},
    "admiration": {"ring": "intense", "sector": "trust", "color": "1E8449"},
    "terror": {"ring": "intense", "sector": "fear", "color": "145A32"},
    "amazement": {"ring": "intense", "sector": "surprise", "color": "1A5276"},
    "grief": {"ring": "intense", "sector": "sadness", "color": "1B2631"},
    "loathing": {"ring": "intense", "sector": "disgust", "color": "6C3483"},
    "rage": {"ring": "intense", "sector": "anger", "color": "C0392B"},
    "vigilance": {"ring": "intense", "sector": "anticipation", "color": "CA6F1E"},
    # Mild Emotions (Ring 1)
    "serenity": {"ring": "mild", "sector": "joy", "color": "F9E79F"},
    "acceptance": {"ring": "mild", "sector": "trust", "color": "ABEBC6"},
    "apprehension": {"ring": "mild", "sector": "fear", "color": "A9DFBF"},
    "distraction": {"ring": "mild", "sector": "surprise", "color": "AED6F1"},
    "pensiveness": {"ring": "mild", "sector": "sadness", "color": "D6EAF8"},
    "boredom": {"ring": "mild", "sector": "disgust", "color": "E8DAEF"},
    "annoyance": {"ring": "mild", "sector": "anger", "color": "F5CBA7"},
    "interest": {"ring": "mild", "sector": "anticipation", "color": "FDEBD0"},
    # Dyadic Emotions (Blends)
    "optimism": {"ring": "dyadic", "sector": "joy+anticipation", "color": "F0B27A"},
    "love": {"ring": "dyadic", "sector": "joy+trust", "color": "82E0AA"},
    "submission": {"ring": "dyadic", "sector": "trust+fear", "color": "76D7C4"},
    "awe": {"ring": "dyadic", "sector": "fear+surprise", "color": "7FB3D3"},
    "disapproval": {"ring": "dyadic", "sector": "surprise+sadness", "color": "7D9EC0"},
    "remorse": {"ring": "dyadic", "sector": "sadness+disgust", "color": "C39BD3"},
    "contempt": {"ring": "dyadic", "sector": "disgust+anger", "color": "E59866"},
    "aggressiveness": {"ring": "dyadic", "sector": "anger+anticipation", "color": "F1948A"},
}

# Convenience lists
PRIMARY_EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
EMOTION_NAMES = sorted(PLUTCHIK.keys())
NUM_EMOTIONS = len(PLUTCHIK)  # 32

# Ring intensity mapping (used by preprocessing and model)
RING_INTENSITY = {
    "intense": 1.0,
    "primary": 0.5,
    "mild": 0.25,
    "dyadic": 0.6,
}

# Sector opposites (for Plutchik wheel distance metric)
SECTOR_OPPOSITES = {
    "joy": "sadness",
    "trust": "disgust",
    "fear": "anger",
    "surprise": "anticipation",
    "sadness": "joy",
    "disgust": "trust",
    "anger": "fear",
    "anticipation": "surprise",
}


# ============== PLUTCHIK WHEEL DISTANCE METRIC ==============
# 32x32 precomputed distance matrix.
# Distance = 0.85 * angular_distance / pi + 0.15 * ring_distance / 2
#   angular_distance: shortest path around the 8-sector wheel (sectors are 45° apart)
#   ring_distance: |ring_A - ring_B| (mild=0, primary=1, dyadic=1.5, intense=2)
# Cross-sector errors (rage→joy) penalized ~5× more than same-sector intensity errors.

_RING_ORDER = {"mild": 0, "primary": 1, "dyadic": 1.5, "intense": 2}
_SECTOR_ORDER = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
_SECTOR_IDX = {s: i for i, s in enumerate(_SECTOR_ORDER)}
_N_SECTORS = 8


def _angular_distance(s1: str, s2: str) -> float:
    i1, i2 = _SECTOR_IDX[s1], _SECTOR_IDX[s2]
    diff = abs(i1 - i2)
    diff = min(diff, _N_SECTORS - diff)
    return diff / (_N_SECTORS / 2)


def _ring_distance(r1: str, r2: str) -> float:
    return abs(_RING_ORDER[r1] - _RING_ORDER[r2]) / 2.0


def wheel_distance(e1: str, e2: str) -> float:
    """Compute Plutchik wheel distance between two emotions.  Returns float in [0, 1]."""
    if e1 == e2:
        return 0.0
    if e1 not in PLUTCHIK or e2 not in PLUTCHIK:
        return 1.0
    a = _angular_distance(PLUTCHIK[e1]["sector"].split("+")[0], PLUTCHIK[e2]["sector"].split("+")[0])
    r = _ring_distance(PLUTCHIK[e1]["ring"], PLUTCHIK[e2]["ring"])
    return 0.85 * a + 0.15 * r


# Precompute full 32x32 matrix
WHEEL_DISTANCE_MATRIX = torch.zeros(NUM_EMOTIONS, NUM_EMOTIONS, dtype=torch.float32)
for _i, _e1 in enumerate(EMOTION_NAMES):
    for _j, _e2 in enumerate(EMOTION_NAMES):
        WHEEL_DISTANCE_MATRIX[_i, _j] = wheel_distance(_e1, _e2)


def wheel_distance_weight(pred_idx: int, true_idx: int, alpha: float = 2.0) -> float:
    """Return exp(alpha * d(pred, true)) — scaling factor for misclassification loss."""
    return float(torch.exp(alpha * WHEEL_DISTANCE_MATRIX[pred_idx, true_idx]).item())
