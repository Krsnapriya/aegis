/**
 * Gmail Overlay - Plutchik Emotion Detector
 * Adds emotion indicators to Gmail messages
 */

(function() {
  'use strict';

  const analysisCache = new Map();
  const ANALYZED_CLASS = 'plutchik-analyzed';

  /**
   * Extract email body text
   */
  function extractEmailText(emailEl) {
    // Gmail uses various selectors for email content
    const selectors = [
      '.a3s.aiL',           // Main email body
      '[class*="message-body"]',
      '.im',                 // Email content div
      '[data-tooltip="Message body"]'
    ];
    
    for (const selector of selectors) {
      const el = emailEl.querySelector(selector);
      if (el && el.textContent.trim().length > 0) {
        return el.textContent.trim();
      }
    }
    
    // Fallback: get all text content
    return emailEl.textContent.trim();
  }

  /**
   * Create emotion indicator for email
   */
  function createEmotionIndicator(result) {
    const container = document.createElement('div');
    container.className = 'plutchik-indicator plutchik-email-indicator';
    container.title = `${result.emotion} (${result.confidence.toFixed(1)}) - ${result.intensity}\nSarcasm: ${result.sarcasm ? '⚠️ Yes' : 'No'}`;
    
    const dot = document.createElement('span');
    dot.className = 'plutchik-dot';
    dot.style.backgroundColor = result.indicator.dotColor;
    
    if (result.sarcasm && result.sarcasm_score > 0.7) {
      dot.classList.add('plutchik-sarcastic');
    }
    
    container.appendChild(dot);
    
    container.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      showEmotionDetails(container, result);
    });
    
    return container;
  }

  /**
   * Show detailed emotion breakdown
   */
  function showEmotionDetails(anchorEl, result) {
    document.querySelectorAll('.plutchik-popup').forEach(el => el.remove());
    
    const popup = document.createElement('div');
    popup.className = 'plutchik-popup';
    
    const top5 = Object.entries(result.all_emotions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    popup.innerHTML = `
      <div class="plutchik-popup-header">
        <strong>${result.emotion}</strong>
        <span class="plutchik-confidence">${(result.confidence * 100).toFixed(0)}%</span>
      </div>
      <div class="plutchik-popup-body">
        <div class="plutchik-stat">
          <span>Intensity:</span>
          <span class="plutchik-intensity-${result.intensity}">${result.intensity}</span>
        </div>
        <div class="plutchik-stat">
          <span>Sarcasm:</span>
          <span class="${result.sarcasm ? 'plutchik-warning' : ''}">
            ${result.sarcasm ? '⚠️ ' : ''}${(result.sarcasm_score * 100).toFixed(0)}%
          </span>
        </div>
        <hr class="plutchik-divider">
        <div class="plutchik-emotion-bars">
          ${top5.map(([emotion, score]) => `
            <div class="plutchik-bar-row">
              <span class="plutchik-bar-label">${emotion}</span>
              <div class="plutchik-bar-container">
                <div class="plutchik-bar" style="width: ${score * 100}%"></div>
              </div>
              <span class="plutchik-bar-value">${(score * 100).toFixed(0)}%</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
    
    anchorEl.appendChild(popup);
    
    setTimeout(() => {
      document.addEventListener('click', function closePopup(e) {
        if (!popup.contains(e.target) && !anchorEl.contains(e.target)) {
          popup.remove();
          document.removeEventListener('click', closePopup);
        }
      });
    }, 100);
  }

  /**
   * Analyze an email and add indicator
   */
  async function analyzeEmail(emailEl) {
    if (emailEl.classList.contains(ANALYZED_CLASS)) return;
    
    const text = extractEmailText(emailEl);
    if (!text || text.length < 5) return;
    
    const cacheKey = text.substring(0, 100);
    if (analysisCache.has(cacheKey)) {
      const result = analysisCache.get(cacheKey);
      const indicator = createEmotionIndicator(result);
      indicator.style.position = 'absolute';
      indicator.style.top = '8px';
      indicator.style.right = '8px';
      emailEl.style.position = 'relative';
      emailEl.appendChild(indicator);
      emailEl.classList.add(ANALYZED_CLASS);
      return;
    }
    
    chrome.runtime.sendMessage(
      { type: 'ANALYZE_TEXT', text: text },
      (response) => {
        if (response && response.success) {
          analysisCache.set(cacheKey, response);
          const indicator = createEmotionIndicator(response);
          indicator.style.position = 'absolute';
          indicator.style.top = '8px';
          indicator.style.right = '8px';
          emailEl.style.position = 'relative';
          emailEl.appendChild(indicator);
        }
      }
    );
    
    emailEl.classList.add(ANALYZED_CLASS);
  }

  /**
   * Find and analyze visible emails
   */
  function analyzeVisibleEmails() {
    const emails = document.querySelectorAll('.a3s.aiL, .im, [role="listitem"]');
    
    emails.forEach(email => {
      if (!email.classList.contains(ANALYZED_CLASS) && email.offsetWidth > 200) {
        analyzeEmail(email);
      }
    });
  }

  function debounce(func, wait) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  }

  console.log('[Plutchik] Gmail overlay loaded');
  
  setTimeout(analyzeVisibleEmails, 1500);
  
  const observer = new MutationObserver(debounce(analyzeVisibleEmails, 500));
  observer.observe(document.body, { childList: true, subtree: true });
  
})();
