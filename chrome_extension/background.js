// Plutchik Dynamic Coach - Background Service Worker
// Handles on-demand activation and multi-user request routing

let activeSessions = new Map();
let requestIdCounter = 0;

// Create context menu items on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'plutchik-analyze',
    title: 'Analyze with Plutchik',
    contexts: ['selection']
  });
  
  console.log('[Plutchik] Extension installed. Ready for on-demand activation.');
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'plutchik-analyze' && info.selectionText) {
    // Send message to content script to analyze selected text
    chrome.tabs.sendMessage(tab.id, {
      action: 'analyze',
      text: info.selectionText
    }, (response) => {
      if (chrome.runtime.lastError || response?.status === 'not_active') {
        // Extension not activated yet, show notification
        showActivationNotification(tab.id);
      }
    });
  }
});

// Handle extension icon click (activation)
chrome.action.onClicked.addListener((tab) => {
  // Activate content script on current tab
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content_script.js']
  }, () => {
    // Send activation message
    chrome.tabs.sendMessage(tab.id, { action: 'activate' }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('[Plutchik] Activation failed:', chrome.runtime.lastError);
      } else {
        console.log('[Plutchik] Activated in tab:', tab.id);
      }
    });
  });
});

// Handle keyboard commands
chrome.commands.onCommand.addListener((command) => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      if (command === '_execute_action') {
        // Activate extension
        chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          files: ['content_script.js']
        }, () => {
          chrome.tabs.sendMessage(tabs[0].id, { action: 'activate' });
        });
      } else if (command === 'analyze-selection') {
        // Analyze selected text
        chrome.tabs.sendMessage(tabs[0].id, { 
          action: 'trigger-selection-analysis' 
        });
      }
    }
  });
});

// Show notification when user tries to use extension before activation
function showActivationNotification(tabId) {
  chrome.tabs.sendMessage(tabId, {
    action: 'show-activation-prompt'
  });
}

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'register-session') {
    // Register a new analysis session
    const sessionId = `session_${Date.now()}_${++requestIdCounter}`;
    activeSessions.set(sessionId, {
      id: sessionId,
      tabId: sender.tab?.id,
      createdAt: Date.now(),
      status: 'active'
    });
    
    console.log(`[Plutchik] Session registered: ${sessionId}`);
    sendResponse({ sessionId });
  } 
  else if (request.action === 'complete-session') {
    // Mark session as complete
    const session = activeSessions.get(request.sessionId);
    if (session) {
      session.status = 'completed';
      session.completedAt = Date.now();
    }
    sendResponse({ success: true });
  }
  else if (request.action === 'get-active-sessions') {
    // Return count of active sessions (for debugging)
    const activeCount = Array.from(activeSessions.values())
      .filter(s => s.status === 'active').length;
    sendResponse({ count: activeCount });
  }
  
  return true; // Keep channel open for async response
});

// Cleanup old sessions periodically
setInterval(() => {
  const now = Date.now();
  const maxAge = 5 * 60 * 1000; // 5 minutes
  
  for (const [id, session] of activeSessions.entries()) {
    if (now - session.createdAt > maxAge) {
      activeSessions.delete(id);
      console.log(`[Plutchik] Cleaned up stale session: ${id}`);
    }
  }
}, 60 * 1000); // Run every minute

console.log('[Plutchik] Background service worker initialized');
