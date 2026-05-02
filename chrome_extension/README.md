# Plutchik Emotion Detector - Chrome Extension

Real-time emotion detection for online conversations with 32-class Plutchik emotions, sarcasm detection, and intensity rings.

## Features

### 🎨 32-Class Emotion Detection
- Full Plutchik wheel coverage: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- Plus dyadic emotions: love, contempt, remorse, awe, submission, aggressiveness, optimism, disapproval
- Intensity rings: mild → primary → intense

### ⚠️ Sarcasm Detection
- Golden ring animation on sarcastic comments
- Thread-level sarcasm rate summary
- Professional tone warnings on LinkedIn

### 🔒 Privacy-First Architecture
- On-device inference (WebGPU/WASM in production)
- No text leaves your browser
- Local caching for performance

## Installation

### Development Mode

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `/workspace/chrome_extension` folder
5. Extension icon appears in toolbar

### Production Build

```bash
# Zip for Chrome Web Store submission
cd /workspace/chrome_extension
zip -r plutchik-emotion-detector.zip \
  manifest.json \
  background.js \
  popup.html \
  popup.js \
  reddit-overlay.js \
  gmail-overlay.js \
  linkedin-overlay.js \
  styles.css \
  icons/
```

## Supported Sites

| Site | Feature | Selector |
|------|---------|----------|
| Reddit | Comment emotion dots | `[data-testid="comment"]` |
| Gmail | Email body indicators | `.a3s.aiL`, `.im` |
| LinkedIn | Professional tone warnings | `.comments-comment`, `.feed-shared-update-v2` |

## How It Works

### Passive Overlay (Default)
- Small colored dot appears on each comment/email
- **Red ring** = intense emotion
- **Green ring** = primary emotion  
- **Blue ring** = mild emotion
- **Golden dashed ring** = sarcasm detected

### Click for Details
Click any dot to see:
- Top 5 emotion distribution
- Confidence score
- Intensity level
- Sarcasm probability
- Professional tone warnings (LinkedIn only)

### Popup Analyzer
Click extension icon to:
- Analyze custom text
- Check model connection status
- View full emotion breakdown

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ Content Scripts │────▶│  Background  │────▶│   API or    │
│ (Reddit/Gmail/  │     │   Service    │     │  WASM Model │
│   LinkedIn)     │◀────│   Worker     │◀────│             │
└─────────────────┘     └──────────────┘     └─────────────┘
        │                      │
        ▼                      ▼
┌─────────────────┐     ┌──────────────┐
│  Emotion Dots   │     │   Popup UI   │
│  + Tooltips     │     │              │
└─────────────────┘     └──────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `manifest.json` | Extension configuration (Manifest V3) |
| `background.js` | Service worker, model inference |
| `reddit-overlay.js` | Reddit comment analysis |
| `gmail-overlay.js` | Gmail email analysis |
| `linkedin-overlay.js` | LinkedIn professional tone advisor |
| `popup.html` | Extension popup UI |
| `popup.js` | Popup logic |
| `styles.css` | Shared styles |
| `icons/` | Extension icons (16/48/128px) |

## Configuration

### API Endpoint (Development)
By default, connects to `http://localhost:8000/predict`. 

To use a different endpoint, edit `background.js`:
```javascript
const API_ENDPOINT = 'https://your-api.com/predict';
```

### Production Mode
In production, the extension loads an ONNX model via WebGPU:
```javascript
// background.js line 60-70
const session = await ort.InferenceSession.create('model.onnx');
```

## Privacy

- ✅ No text sent to external servers (production)
- ✅ All inference runs locally
- ✅ Cache stored in memory only
- ✅ No tracking or analytics

## Troubleshooting

### "Model not ready" error
- Ensure inference server is running: `python inference_server.py`
- Check console for errors: `Ctrl+Shift+J` → Console tab

### No emotion dots appearing
- Refresh the page after installing extension
- Check if site is in `host_permissions` in manifest.json
- Verify content script loaded: look for `[Plutchik] Reddit overlay loaded` in console

### Icons not showing
- Run `python3 icons/generate_icons.py` to regenerate PNG files

## Development

### Testing Locally

1. Start inference server:
```bash
cd /workspace
python inference_server.py
```

2. Load extension in Chrome (see Installation)

3. Visit `reddit.com` and scroll to load comments

4. Open DevTools console to see debug logs

### Adding New Sites

1. Create `newsite-overlay.js`:
```javascript
(function() {
  function extractText(el) { /* ... */ }
  function analyzeVisibleContent() { /* ... */ }
  
  // Observe DOM and call analyzeVisibleContent
})();
```

2. Add to `manifest.json`:
```json
{
  "matches": ["https://www.newsite.com/*"],
  "js": ["newsite-overlay.js"],
  "css": ["styles.css"]
}
```

## License

MIT License - See LICENSE file

## Credits

Based on Plutchik's Wheel of Emotions (1980)
32-class emotion recognition model trained on Plutchik ERC dataset
