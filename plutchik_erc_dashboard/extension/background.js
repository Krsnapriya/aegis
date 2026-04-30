/**
 * Plutchik Emotion Lens — Background Service Worker
 */

let API_BASE = 'http://localhost:8000'; // Default
chrome.storage.local.get(['endpoint'], (result) => {
  if (result.endpoint) API_BASE = result.endpoint;
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'PREDICT_EMOTION') {
    fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: request.text,
        session_id: request.sessionId || 'browser-extension',
        scenario: request.scenario || 'social',
        topic: request.topic || 'general'
      })
    })
      .then(response => response.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    
    return true; // Keep message channel open for async response
  }
});
