# Plutchik Dynamic Coach - Privacy-First Chrome Extension

## 🚀 Quick Start

### Installation (Development)

1. **Start the Inference Server** (optional, for cloud mode):
   ```bash
   cd /workspace
   python inference_server.py
   ```

2. **Export Model for On-Device Inference** (recommended for privacy):
   ```bash
   python export_for_browser.py
   ```
   This creates ONNX models in `chrome_extension/model/` for 100% browser-based processing.

3. **Load Extension in Chrome**:
   - Open Chrome → Navigate to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top-right)
   - Click "Load unpacked"
   - Select the `/workspace/chrome_extension` folder
   - Extension icon appears in toolbar

4. **Verify Installation**:
   - Click extension icon → Should show "Activate Plutchik Coach"
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → Floating indicator appears
   - ✅ Extension is ready!

---

## 🔒 Privacy-First by Design

### Default Behavior: 100% On-Device Processing

| Feature | Implementation |
|---------|----------------|
| **Inference** | WASM/WebGPU in browser (no network calls) |
| **Data Storage** | Ephemeral only (5-minute session expiry) |
| **Tracking** | Zero analytics, telemetry, or cookies |
| **Permissions** | Minimal (activeTab, storage for preferences only) |
| **Activation** | User-triggered only (inactive by default) |

**See [PRIVACY_ARCHITECTURE.md](./PRIVACY_ARCHITECTURE.md) for complete details.**

---

## 🎯 How to Use

### Method 1: Keyboard Shortcut (Fastest)

1. Navigate to Reddit, Gmail, or LinkedIn
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
3. Floating indicator "✨ Plutchik Active" appears
4. Click in any text box → "🔍 Analyze Tone" button appears
5. Click button → Analysis panel shows emotions, sarcasm, trajectory

### Method 2: Extension Icon

1. Click Plutchik icon in Chrome toolbar
2. Extension activates on current tab
3. Same as above: analyze buttons appear in text boxes

### Method 3: Text Selection

1. Highlight any text on the page
2. Tooltip appears: "🔍 Analyze with Plutchik"
3. Click tooltip → Instant analysis

### Method 4: Right-Click Context Menu

1. Select text
2. Right-click → "Analyze with Plutchik"
3. Analysis panel appears

---

## 📊 Features

### Real-Time Emotional Intelligence

When you analyze text, you get:

1. **Primary Emotion Detection** (32-class Plutchik model)
   - Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
   - Plus 24 nuanced blends (Love, Contempt, Remorse, Awe, etc.)

2. **Sarcasm Probability**
   - Detects passive-aggression and incongruity
   - Flags likely sarcasm (>30% probability)

3. **Intensity Level**
   - Low / Medium / High intensity rings
   - Helps gauge emotional temperature

4. **Trajectory Forecast** (Cloud mode only)
   - Predicts where conversation is heading emotionally
   - Identifies inflection points

5. **Reframe Suggestions** (Cloud mode only)
   - Alternative phrasings to de-escalate tension
   - Click to copy suggested rewrites

6. **Baseline Deviation** (Opt-in)
   - Compares current text to your historical patterns
   - Alerts when you're acting out of character

---

## 🌐 Supported Sites

| Site | Features | Notes |
|------|----------|-------|
| **Reddit** | Full support | Works on comments, posts, DMs |
| **Gmail** | Full support | Compose, reply, forward |
| **LinkedIn** | Full support | Posts, comments, messages |
| **Other sites** | Basic support | Text selection + context menu work everywhere |

---

## ⚙️ Settings

Click extension icon → Gear icon (Settings)

### Privacy Settings

- **On-Device Mode** (Default): All processing in browser
- **Cloud Mode** (Opt-in): Advanced features via API
  - Requires API key from `localhost:8000` or cloud provider
  - Review data handling disclosure before enabling

### Personalization

- **Enable Baseline Tracking** (Off by default)
  - Stores your emotional patterns locally
  - Provides "unusual for you" alerts
  - Can be cleared anytime with "Reset Baseline" button

### Site Permissions

- Toggle support for Reddit, Gmail, LinkedIn individually
- Add custom sites (advanced users)

---

## 🛠️ Architecture

### Components

```
chrome_extension/
├── manifest.json          # Extension configuration
├── background.js          # Service worker (session management)
├── content_script.js      # UI injection & user interaction
├── ondevice-inference.js  # ONNX runtime wrapper (browser-based)
├── reddit-overlay.js      # Reddit-specific integrations
├── gmail-overlay.js       # Gmail-specific integrations
├── linkedin-overlay.js    # LinkedIn-specific integrations
├── popup.html             # Settings popup UI
├── popup.js               # Popup logic
├── styles.css             # Shared styles
├── model/                 # ONNX models (after export)
│   ├── emotion_head.onnx
│   ├── sarcasm_head.onnx
│   ├── intensity_head.onnx
│   └── config.json
└── PRIVACY_ARCHITECTURE.md # Detailed privacy documentation
```

### Request Flow (On-Device Mode)

```
User clicks "Analyze Tone"
    ↓
content_script.js captures text
    ↓
ondevice-inference.js tokenizes input
    ↓
ONNX models run in browser (WebGPU/WASM)
    ↓
Results formatted and displayed
    ↓
Session auto-deletes after 5 minutes
```

**Zero network requests. Zero data leaves device.**

### Request Flow (Cloud Mode - Opt-In)

```
User clicks "Analyze Tone"
    ↓
content_script.js captures text
    ↓
background.js generates unique session_id
    ↓
HTTPS POST to API endpoint
    ↓
Server processes and returns results
    ↓
Results displayed to user
    ↓
Session deleted after response (or 5 min timeout)
```

**Encrypted transit. No logging of text content.**

---

## 🧪 Testing

### Verify On-Device Processing

1. **Disconnect from Internet**
   - Turn off WiFi / unplug Ethernet
   
2. **Activate Extension**
   - Press `Ctrl+Shift+P`
   
3. **Analyze Text**
   - Type in any text box → Click "Analyze Tone"
   
4. **Expected Result**
   - ✅ Analysis completes successfully
   - ✅ No error messages
   - ✅ Proves 100% browser-based processing

### Verify No Network Traffic

1. Open Chrome DevTools → Network tab
2. Clear network log
3. Activate extension and analyze text
4. **Expected:** Zero network requests (in on-device mode)

### Verify Session Cleanup

1. Open Chrome DevTools → Console
2. Activate extension and analyze text multiple times
3. Wait 5 minutes
4. Check console logs → Should see "Cleaned up stale session" messages

---

## 🔐 Multi-User Support

### How Simultaneous Users Stay Isolated

Each analysis request gets a unique identifier:

```javascript
const sessionId = `req_${Date.now()}_${randomCounter}`;
// Example: req_1704892341_7
```

- **No shared state** between tabs or users
- **Ephemeral sessions** (5-minute expiry)
- **Automatic cleanup** via background service worker

### Example Scenario

```
User A (Tab 1) → req_1704892341_1 → Analysis → Display → Deleted
User B (Tab 2) → req_1704892342_1 → Analysis → Display → Deleted
User A (Tab 3) → req_1704892350_2 → Analysis → Display → Deleted
```

**No cross-contamination possible.**

---

## 🚨 Troubleshooting

### "Analysis Failed" Error

**Cause:** Extension trying to reach cloud API but server not running

**Solution:**
1. Check if you're in Cloud Mode (Settings → Cloud Processing)
2. Either:
   - Start server: `python inference_server.py`
   - OR switch to On-Device Mode (recommended)

### Model Not Loading

**Cause:** ONNX models not exported yet

**Solution:**
```bash
python export_for_browser.py
```

Then reload extension in `chrome://extensions/`

### Buttons Not Appearing

**Cause:** Extension not activated

**Solution:**
- Press `Ctrl+Shift+P` or click extension icon
- Look for "✨ Plutchik Active" floating indicator

### Slow Performance

**Cause:** WebGPU not available, falling back to WASM

**Solution:**
- Update Chrome to latest version
- Ensure hardware acceleration is enabled
- Consider cloud mode for faster inference (opt-in)

---

## 📦 Development

### Building from Source

1. **Install Dependencies**:
   ```bash
   pip install torch onnx onnxruntime
   ```

2. **Export Models**:
   ```bash
   python export_for_browser.py
   ```

3. **Load Unpacked Extension**:
   - See installation steps above

### Making Changes

- Edit JavaScript files → Reload extension in `chrome://extensions/`
- Edit Python server → Restart `inference_server.py`
- Edit models → Re-run `export_for_browser.py` → Reload extension

### Debugging

Open DevTools for extension background page:
1. Go to `chrome://extensions/`
2. Find Plutchik Dynamic Coach
3. Click "Inspect views: background page"
4. Console logs appear

---

## 📄 License & Credits

- **License:** MIT (open source)
- **Model:** Plutchik ERC 32-class emotion detection
- **Privacy:** Zero-knowledge architecture
- **Created by:** Plutchik ERC Team

For questions: privacy@plutchik-erc.dev

---

## 🔮 Roadmap

### Q1 2025
- [ ] Firefox extension support
- [ ] Safari extension support
- [ ] Custom site configurations

### Q2 2025
- [ ] Federated learning for personalization
- [ ] Differential privacy for aggregate stats
- [ ] Mobile keyboard extension (iOS/Android)

### Q3 2025
- [ ] Multimodal analysis (emoji, images)
- [ ] Real-time conversation coaching (voice calls)
- [ ] Team/enterprise dashboard (opt-in)

---

**Last Updated:** January 2025  
**Version:** 3.0.0 (Privacy-First Release)
