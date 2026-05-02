// Plutchik Dynamic Coach - Background Service Worker (MV3, type: module)
// Handles on-demand activation, on-device inference, and ephemeral session management.
// Privacy guarantee: text is processed entirely in-browser. No data leaves the device.

import PlutchikOnDeviceInference from './ondevice-inference.js';

// ─── Inference engine (lazy singleton) ────────────────────────────────────────
// The service worker may be killed and restarted by Chrome; the engine is
// re-initialised on the first message after a restart.
let _enginePromise = null;

function getEngine() {
  if (!_enginePromise) {
    const engine = new PlutchikOnDeviceInference();
    _enginePromise = engine.initialize().then(() => engine);
  }
  return _enginePromise;
}

// ─── Ephemeral session store ──────────────────────────────────────────────────
// Sessions are held only in-memory and auto-expire after 5 minutes.
// Format: req_{timestamp}_{randomHex}   (spec: req_{timestamp}_{uuid})
const activeSessions = new Map();

function generateRequestId() {
  const ts  = Date.now();
  const rnd = typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
    : Math.random().toString(36).slice(2, 14);
  return `req_${ts}_${rnd}`;
}

// Cleanup sessions older than 5 minutes
setInterval(() => {
  const now    = Date.now();
  const maxAge = 5 * 60 * 1000;
  for (const [id, session] of activeSessions.entries()) {
    if (now - session.createdAt > maxAge) {
      activeSessions.delete(id);
      console.log(`[Plutchik] Cleaned up stale session: ${id}`);
    }
  }
}, 60 * 1000);

// ─── Extension install ────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'plutchik-analyze',
    title: 'Analyze with Plutchik',
    contexts: ['selection']
  });
  console.log('[Plutchik] Extension installed. Ready for on-demand activation.');
});

// ─── Context menu ─────────────────────────────────────────────────────────────

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'plutchik-analyze' && info.selectionText) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'analyze',
      text: info.selectionText
    }, (response) => {
      if (chrome.runtime.lastError || response?.status === 'not_active') {
        chrome.tabs.sendMessage(tab.id, { action: 'show-activation-prompt' });
      }
    });
  }
});

// ─── Icon click → activate content script ────────────────────────────────────

chrome.action.onClicked.addListener((tab) => {
  chrome.scripting.executeScript(
    { target: { tabId: tab.id }, files: ['content_script.js'] },
    () => {
      if (chrome.runtime.lastError) {
        console.error('[Plutchik] Script injection failed:', chrome.runtime.lastError);
        return;
      }
      chrome.tabs.sendMessage(tab.id, { action: 'activate' }, (resp) => {
        if (!chrome.runtime.lastError) {
          console.log('[Plutchik] Activated in tab:', tab.id);
        }
      });
    }
  );
});

// ─── Keyboard commands ────────────────────────────────────────────────────────

chrome.commands.onCommand.addListener((command) => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;
    const tabId = tabs[0].id;

    if (command === '_execute_action') {
      chrome.scripting.executeScript(
        { target: { tabId }, files: ['content_script.js'] },
        () => { chrome.tabs.sendMessage(tabId, { action: 'activate' }); }
      );
    } else if (command === 'analyze-selection') {
      chrome.tabs.sendMessage(tabId, { action: 'trigger-selection-analysis' });
    }
  });
});

// ─── Message handler ──────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

  // ── On-device analysis (privacy-first default) ──
  if (request.action === 'analyzeText') {
    const requestId = generateRequestId();
    activeSessions.set(requestId, {
      id: requestId,
      tabId: sender.tab?.id,
      createdAt: Date.now(),
      status: 'active'
    });
    console.log(`[Plutchik] Analysis request: ${requestId}`);

    getEngine()
      .then(engine => engine.analyze(request.text))
      .then(result => {
        const session = activeSessions.get(requestId);
        if (session) { session.status = 'completed'; session.completedAt = Date.now(); }
        sendResponse({ success: true, data: result, requestId });
      })
      .catch(err => {
        console.error('[Plutchik] Inference error:', err);
        sendResponse({ success: false, error: err.message });
      });

    return true; // keep message channel open for async sendResponse
  }

  // ── Session management helpers ──
  if (request.action === 'register-session') {
    const sessionId = generateRequestId();
    activeSessions.set(sessionId, {
      id: sessionId,
      tabId: sender.tab?.id,
      createdAt: Date.now(),
      status: 'active'
    });
    sendResponse({ sessionId });
    return true;
  }

  if (request.action === 'complete-session') {
    const session = activeSessions.get(request.sessionId);
    if (session) { session.status = 'completed'; session.completedAt = Date.now(); }
    sendResponse({ success: true });
    return true;
  }

  if (request.action === 'get-active-sessions') {
    const count = Array.from(activeSessions.values()).filter(s => s.status === 'active').length;
    sendResponse({ count });
    return true;
  }

  // ── Inference engine status (for popup) ──
  if (request.action === 'get-engine-status') {
    getEngine()
      .then(engine => {
        sendResponse({
          ready: engine.initialized,
          mode: engine.useONNX ? 'onnx' : 'heuristic',
          privacy: 'on_device'
        });
      })
      .catch(() => sendResponse({ ready: false, mode: 'unavailable', privacy: 'on_device' }));
    return true;
  }
});

console.log('[Plutchik] Background service worker initialised (on-device, privacy-first)');
