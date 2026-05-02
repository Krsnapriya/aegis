# Implementation Complete: Plutchik ERC Dynamic Privacy-First Extension

## ✅ What Is Implemented

### 1. Privacy-First Architecture (Default: 100% On-Device)

| Layer | Status | Detail |
|---|---|---|
| Default inference | ✅ | On-device ONNX/WebAssembly via `ondevice-inference.js` |
| Server calls | ✅ Removed | `content_script.js` and `popup.js` make **zero** remote calls by default |
| localhost permissions | ✅ Removed | Removed from `manifest.json` `host_permissions` |
| Ephemeral sessions | ✅ | `req_{timestamp}_{uuid}` format, auto-expire 5 min |
| No telemetry | ✅ | Zero analytics, cookies, or tracking |
| Heuristic fallback | ✅ | Works without ONNX models (pure-JS lexicon engine) |

### 2. Chrome Extension (Manifest V3)

#### `manifest.json`
- MV3, service worker with `"type": "module"`
- Keyboard shortcut `Ctrl+Shift+P` (via `_execute_action`)
- Host permissions: Reddit, Gmail, LinkedIn only (no localhost by default)
- `web_accessible_resources`: icons, styles, model/, lib/, ondevice-inference.js

#### `background.js`
- Imports `PlutchikOnDeviceInference` from `ondevice-inference.js` (ES module)
- Lazy-initialises inference engine (survives service-worker restarts)
- Handles `analyzeText` messages with async response
- Session management: `req_{timestamp}_{uuid}`, 5-minute cleanup

#### `content_script.js`
- On-demand activation via `activate` message
- Shows "✨ Plutchik Active" floating indicator on activation
- Scans for `textarea`, `contenteditable`, `input[type=text]` and injects "🔍 Analyze Tone" button
- Delegates all inference to background via `chrome.runtime.sendMessage`
- Shows slide-out panel with: risk level, emotion, sarcasm flag, trajectory, 3 reframe buttons

#### `popup.js`
- Shows engine status (ONNX or heuristic, on-device)
- Analysis via background message (no direct API call)

#### `ondevice-inference.js`
- ES module, imported by background service worker
- **ONNX path** (when `lib/ort.esm.min.js` is present): full ONNX inference via WebGPU/WASM
- **Heuristic fallback** (always available, no download required):
  - 32-class emotion classification via word-emotion lexicon
  - Multimodal incongruity detection (sarcasm/passive-aggression)
  - Neural-ODE-inspired trajectory forecast (10 steps, mean-reversion dynamics)
  - Counterfactual reframe generation (3 Trust/Serenity-targeting alternatives)

### 3. Advanced Algorithms (`advanced_engine.py`)

#### A. Neural ODE Trajectory Forecaster
- `EmotionODEFunc` + `TrajectoryForecaster`
- RK4 integration via `torchdiffeq`
- 10-step forecast with inflection-point detection
- Risk score from emotional volatility

#### B. Multimodal Incongruity Detector
- `MultimodalIncongruityDetector`
- Caps ratio, exclamation count, positive/negative lexical contrast
- Detects sarcasm and passive-aggression

#### C. Counterfactual Reframe Generator
- `CounterfactualGenerator`
- Template-based perturbations targeting Trust/Serenity
- 3 alternatives preserving core message intent
- TODO: replace with T5/GPT neural reframing

#### D. Unified Engine
- `AdvancedPlutchikEngine.analyze_dynamic()` combines all three
- Used by `inference_server.py` (optional server mode)

### 4. Python Backend (`inference_server.py`) — Optional

> **Not required for normal extension use.** Default mode is 100% on-device.

- FastAPI server with `/predict`, `/analyze/dynamic`, `/predict/arc`, etc.
- Configurable model path: `--model`, `$PLUTCHIK_MODEL_PATH`, or repo-relative default
- Corrections logged to `data/corrections.jsonl` (repo-relative, not `/workspace/`)
- CORS enabled (restrict origins before production deployment)

### 5. Model Export (`export_for_browser.py`)

- Exports `encoder.onnx`, `emotion_head.onnx`, `sarcasm_head.onnx`, `intensity_head.onnx`
- Saves `model/config.json` with emotion class list
- Configurable model path: `sys.argv[1]`, `$PLUTCHIK_MODEL_PATH`, or repo-relative default

### 6. Validation (`validate.py`)

```bash
python validate.py
```
Checks: imports, model forward pass, advanced engine, manifest privacy requirements,
extension file correctness.

---

## 🚀 Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Optional) Train model
python scripts/train.py

# 3. Export ONNX models for browser
python export_for_browser.py

# 4. (Optional) Download onnxruntime-web ESM build
#    See chrome_extension/lib/INSTALL.md

# 5. Load extension in Chrome
#    chrome://extensions → Developer mode → Load unpacked → select chrome_extension/

# 6. Run validation
python validate.py

# 7. (Optional) Start local server for cloud/advanced features
python inference_server.py
```

---

## 📂 File Status

| File | Purpose | Status |
|---|---|---|
| `advanced_engine.py` | Neural ODE, Incongruity, Reframes (Python) | ✅ Fixed timestamp |
| `export_for_browser.py` | ONNX export script | ✅ Fixed hardcoded paths |
| `inference_server.py` | Optional FastAPI server | ✅ Fixed paths, optional-only doc |
| `validate.py` | Local sanity-check script | ✅ New |
| `requirements.txt` | Python deps | ✅ Added onnx, onnxruntime |
| `chrome_extension/manifest.json` | MV3 manifest | ✅ Removed localhost, fixed resources |
| `chrome_extension/background.js` | Service worker | ✅ Imports on-device engine |
| `chrome_extension/content_script.js` | DOM injection + UI | ✅ No API calls, on-device only |
| `chrome_extension/ondevice-inference.js` | ONNX + heuristic engine | ✅ Complete rewrite |
| `chrome_extension/popup.js` | Popup UI | ✅ No API calls |
| `chrome_extension/lib/INSTALL.md` | ort.js download instructions | ✅ New |
| `chrome_extension/PRIVACY_ARCHITECTURE.md` | Privacy docs | ✅ Accurate |
| `chrome_extension/README.md` | User docs | ✅ Accurate |
| `models/multitask_emotion_model.py` | Tiny BERT model | ✅ Unchanged |
| `scripts/train.py` | Training script | Unchanged |

---

## 🔒 Privacy Guarantees

- ✅ Default: zero network requests during analysis
- ✅ Session IDs format: `req_{timestamp}_{uuid}`
- ✅ Sessions expire after 5 minutes (background cleanup)
- ✅ `manifest.json` does not include localhost in `host_permissions`
- ✅ No `chrome.storage` used for analyzed text
- ✅ Optional server mode requires manual opt-in (not in extension UI by default)

---

## ⚠️ Known TODOs / Limitations

1. **Tokeniser**: `ondevice-inference.js` uses a simple hash tokeniser.
   Replace with a proper WordPiece tokeniser for accurate ONNX inference.
2. **Neural reframes**: `CounterfactualGenerator` uses rule-based templates.
   TODO: fine-tune a T5 model and export to ONNX.
3. **ort.js bundle**: Users must manually download `ort.esm.min.js`
   (see `chrome_extension/lib/INSTALL.md`). A future build step could automate this.
4. **Model quantisation**: The BERT encoder may be ~5 MB; INT8 quantisation
   can reduce it to ~2 MB for faster loading.
5. **Test coverage**: Unit tests exist for Python components; browser-side
   automated tests are not yet implemented.

---

**Last Updated:** May 2026  
**Version:** 3.0.0 (Privacy-First, On-Device-Default)

