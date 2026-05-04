/**
 * Plutchik Emotion Lens — Background Service Worker
 */

let API_BASE = 'http://localhost:8000'; // Default
chrome.storage.local.get(['endpoint'], (result) => {
  if (result.endpoint) API_BASE = result.endpoint;
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'PREDICT_EMOTION') {
    chrome.storage.local.get(['apiKey'], (keys) => {
      const headers = { 'Content-Type': 'application/json' };
      if (keys.apiKey) headers['X-API-Key'] = keys.apiKey;
      fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          text: request.text,
          session_id: request.sessionId || 'browser-extension',
          scenario: request.scenario || 'social',
          topic: request.topic || 'general'
        })
      })
        .then((response) => response.json())
        .then((data) => sendResponse({ success: true, data }))
        .catch((error) => sendResponse({ success: false, error: error.message }));
    });
    return true;
  }
});
