"""
Plutchik ERC v2.1 — Canonical Constants
Single source of truth for the Plutchik emotion taxonomy.
All modules import from here. Do not duplicate this dictionary elsewhere.
"""

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
