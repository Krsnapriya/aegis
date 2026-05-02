# Privacy-First Architecture

## Core Principle: Zero Data Leaves the Device (By Default)

The Plutchik Dynamic Coach extension is built on a **local-first, privacy-by-design** architecture. No text, conversation data, or personal information ever leaves your browser unless you explicitly opt into cloud features.

---

## 🔒 Privacy Guarantees

### 1. **On-Device Inference (Default)**
- ✅ Model runs entirely in your browser using WebAssembly (WASM) and WebGPU
- ✅ Zero network requests for emotion analysis
- ✅ No text sent to servers, APIs, or third parties
- ✅ Works offline after initial installation

### 2. **Ephemeral Session Management**
- ✅ Each analysis request gets a unique, random `session_id`
- ✅ Sessions auto-expire after 5 minutes
- ✅ No persistent storage of analyzed text
- ✅ No cross-tab or cross-site data sharing

### 3. **Minimal Permissions**
- ✅ Only requests permissions essential for functionality
- ✅ No access to browsing history, cookies, or passwords
- ✅ Host permissions limited to Reddit, Gmail, LinkedIn (user-configurable)
- ✅ No background tracking or analytics

### 4. **User-Controlled Activation**
- ✅ Extension is **inactive by default** on page load
- ✅ Activates only when user clicks icon or presses hotkey (`Ctrl+Shift+P`)
- ✅ Floating indicator shows when active (easy to see and deactivate)
- ✅ One-click deactivation removes all UI elements

### 5. **No Tracking or Analytics**
- ✅ No Google Analytics, Mixpanel, or similar tools
- ✅ No error reporting services (Sentry, etc.)
- ✅ No usage telemetry or behavioral tracking
- ✅ No A/B testing frameworks

### 6. **Open Source & Auditable**
- ✅ All code is visible in the extension package
- ✅ No obfuscated or minified scripts hiding behavior
- ✅ Community can audit for privacy violations
- ✅ Build process is reproducible

---

## 🛡️ Technical Implementation

### On-Device Model (WASM/WebGPU)

```javascript
// chrome_extension/ondevice-inference.js
class PlutchikOnDeviceInference {
  async initialize() {
    // Load ONNX models from extension package
    this.emotionSession = await ort.InferenceSession.create(
      chrome.runtime.getURL('model/emotion_head.onnx'),
      { executionProviders: ['webgpu', 'wasm'] }
    );
    // All processing happens locally - no network calls
  }
  
  async analyze(text) {
    // Tokenize and run inference entirely in browser
    const results = await this.emotionSession.run({ input_ids: tensor });
    return results; // Never leaves the device
  }
}
```

### Ephemeral Session Cleanup

```javascript
// chrome_extension/background.js
setInterval(() => {
  const now = Date.now();
  const maxAge = 5 * 60 * 1000; // 5 minutes
  
  for (const [id, session] of activeSessions.entries()) {
    if (now - session.createdAt > maxAge) {
      activeSessions.delete(id); // Auto-cleanup
      console.log(`[Plutchik] Cleaned up stale session: ${id}`);
    }
  }
}, 60 * 1000);
```

### No Persistent Storage

```javascript
// Explicitly NOT using chrome.storage for analyzed text
// Only storing minimal user preferences (opt-in settings)

chrome.storage.local.set({
  // ONLY user preferences, NO analyzed content
  cloudModeEnabled: false,  // Default: false (on-device only)
  enableBaselineTracking: false  // Default: false
});
```

---

## 🔐 Multi-User Isolation

### How Simultaneous Users Stay Isolated

| Feature | Implementation |
|---------|----------------|
| **Session IDs** | Unique per-request: `session_${timestamp}_${randomCounter}` |
| **Tab Isolation** | Each browser tab gets independent analysis context |
| **No Shared State** | Background service worker maintains no global user data |
| **Request Scoping** | All API calls include isolated `session_id` |
| **Auto-Cleanup** | Sessions expire after 5 minutes automatically |

### Example Flow for Multiple Users

```
User A (Tab 1) → Request ID: req_1704892341_1 → Analysis → Response → Session Deleted
User B (Tab 2) → Request ID: req_1704892342_1 → Analysis → Response → Session Deleted
User A (Tab 3) → Request ID: req_1704892350_2 → Analysis → Response → Session Deleted
```

**No data persists between requests. No cross-contamination possible.**

---

## ☁️ Cloud Mode (Opt-In Only)

### When Cloud Processing is Used

By default, **all processing is on-device**. Cloud mode is an **explicit opt-in** feature for users who:
- Want access to advanced features (trajectory forecasting, reframe generation)
- Are on devices without WebGPU support
- Prefer faster inference over maximum privacy

### How to Enable Cloud Mode

1. Click extension icon → Settings
2. Toggle "Enable Cloud Processing" to ON
3. Review and accept data handling disclosure
4. Cloud mode activates (indicated by cloud icon in UI)

### Cloud Data Handling

If cloud mode is enabled:

| Data Type | Retention | Usage |
|-----------|-----------|-------|
| Analyzed text | Not stored | Processed in-memory, discarded after response |
| Session ID | 24 hours | Rate limiting and abuse prevention only |
| API keys | Encrypted | Stored in browser, never transmitted |
| Usage metrics | Aggregated | Anonymous counts only (no text content) |

### Cloud Security Measures

- ✅ HTTPS-only communication (TLS 1.3)
- ✅ API key authentication required
- ✅ Rate limiting per API key
- ✅ No logging of request bodies
- ✅ GDPR-compliant data handling
- ✅ SOC 2 Type II certified infrastructure (if using major cloud providers)

---

## 📋 Permissions Explained

### Manifest.json Permissions

```json
{
  "permissions": [
    "activeTab",        // Only access current active tab (not all tabs)
    "storage",          // Store user preferences locally (not browsing data)
    "scripting",        // Inject content script when activated
    "contextMenus",     // Add right-click menu items
    "commands"          // Keyboard shortcuts (Ctrl+Shift+P)
  ],
  "host_permissions": [
    "https://www.reddit.com/*",   // User-configurable
    "https://mail.google.com/*",  // User-configurable
    "https://www.linkedin.com/*", // User-configurable
    "http://localhost:8000/*"     // Local development only
  ]
}
```

### What We DON'T Request

❌ `tabs` - No access to all open tabs  
❌ `history` - No browsing history access  
❌ `cookies` - No cookie access  
❌ `webRequest` - No network traffic monitoring  
❌ `privacy` - No Chrome privacy settings access  

---

## 🧪 Verification & Transparency

### How to Verify Privacy Claims

1. **Inspect Network Traffic**
   - Open Chrome DevTools → Network tab
   - Activate extension and analyze text
   - **Expected:** Zero network requests (in default on-device mode)

2. **Review Extension Code**
   - Navigate to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" on extension folder
   - Review all `.js` files for network calls

3. **Monitor Storage**
   - Open DevTools → Application → Storage
   - Check `chrome-extension://[ID]/` storage
   - **Expected:** Only user preferences, no analyzed text

4. **Run Offline**
   - Disconnect from internet
   - Activate extension and analyze text
   - **Expected:** Full functionality works (proves on-device processing)

---

## 🚨 What Happens If Privacy is Violated?

### Bug Bounty Program

We offer rewards for verified privacy violations:
- **$500**: Accidental data leakage (unintended network request)
- **$2,000**: Persistent storage of analyzed text without consent
- **$5,000**: Cross-user data contamination
- **$10,000**: Intentional backdoor or tracking mechanism

Report vulnerabilities to: security@plutchik-erc.dev (PGP key available)

---

## 📜 Privacy Policy Summary

### Data We Collect

**None by default.** With cloud mode opt-in:
- API keys (encrypted, stored locally)
- Anonymous usage counts (e.g., "50 analyses performed today")
- Error logs (stack traces only, no text content)

### Data We NEVER Collect

- Analyzed text content
- Conversation history
- User identities or demographics
- Browsing behavior patterns
- Keystroke dynamics
- Mouse movements or clicks

### Data Sharing

**Never.** We do not:
- Sell data to third parties
- Share data with advertisers
- Provide data to data brokers
- Enable government surveillance backdoors

### Data Deletion

- On-device data: Deleted automatically when session expires (5 min)
- Cloud data (if opted in): Delete button in settings wipes all server-side data within 24 hours
- Extension uninstall: All local data removed automatically

---

## 🎯 Comparison with Competitors

| Feature | Plutchik | Competitor A | Competitor B |
|---------|----------|--------------|--------------|
| On-device inference | ✅ Default | ❌ Always cloud | ⚠️ Optional |
| Zero data retention | ✅ Yes | ❌ 30 days | ❌ 90 days |
| Open source | ✅ Fully | ❌ Closed | ⚠️ Partial |
| Opt-in cloud | ✅ Explicit | ❌ Always on | ⚠️ Hidden |
| No tracking | ✅ Guaranteed | ❌ Analytics | ❌ Telemetry |
| Works offline | ✅ Yes | ❌ No | ⚠️ Limited |

---

## 🔮 Future Privacy Enhancements

### Planned (Q2 2025)
- [ ] Federated learning for personalization without data centralization
- [ ] Differential privacy for aggregate statistics
- [ ] Zero-knowledge proofs for model updates
- [ ] Homomorphic encryption for cloud inference (research phase)

### Research Phase
- [ ] Secure multi-party computation for collaborative filtering
- [ ] Trusted execution environments (Intel SGX) for cloud processing
- [ ] Blockchain-based audit trails for data access (controversial)

---

## 📞 Contact

Privacy questions or concerns?
- Email: privacy@plutchik-erc.dev
- PGP Key: Available on website
- Response time: Within 48 hours

---

**Last Updated:** January 2025  
**Version:** 3.0.0 (Privacy-First Release)
