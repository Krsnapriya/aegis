/**
 * LinkedIn Overlay - Plutchik Emotion Detector
 * Professional tone advisor with sarcasm/contempt warnings
 */

(function() {
  'use strict';

  const analysisCache = new Map();
  const ANALYZED_CLASS = 'plutchik-analyzed';

  /**
   * Extract comment/post text from LinkedIn
   */
  function extractText(element) {
    const selectors = [
      '.ember-view',
      '[class*="comment"]',
      '[class*="post"]',
      '.feed-shared-update-v2',
      '.comments-comment'
    ];
    
    for (const selector of selectors) {
      const el = element.querySelector(selector);
      if (el && el.textContent.trim().length > 0) {
        return el.textContent.trim();
      }
    }
    
    return element.textContent.trim();
  }

  /**
   * Create professional tone warning
   */
  function createToneWarning(result) {
    const container = document.createElement('div');
    container.className = 'plutchik-indicator plutchik-linkedin-indicator';
    
    // Special warnings for professional context
    const warnings = [];
    if (result.emotion === 'contempt' && result.confidence > 0.6) {
      warnings.push('⚠️ May read as contempt');
    }
    if (result.sarcasm && result.sarcasm_score > 0.7) {
      warnings.push('⚠️ High sarcasm detected');
    }
    if (result.emotion === 'rage' || result.emotion === 'anger') {
      warnings.push('⚠️ Strong negative emotion');
    }
    if (result.emotion === 'loathing' || result.emotion === 'disgust') {
      warnings.push('⚠️ May seem unprofessional');
    }
    
    const dot = document.createElement('span');
    dot.className = 'plutchik-dot';
    dot.style.backgroundColor = result.indicator.dotColor;
    
    if (warnings.length > 0) {
      dot.classList.add('plutchik-sarcastic');
      dot.title = warnings.join('\n');
    } else {
      dot.title = `${result.emotion} (${result.confidence.toFixed(1)})`;
    }
    
    container.appendChild(dot);
    
    container.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      showEmotionDetails(container, result, warnings);
    });
    
    return container;
  }

  /**
   * Show detailed breakdown with professional advice
   */
  function showEmotionDetails(anchorEl, result, warnings = []) {
    document.querySelectorAll('.plutchik-popup').forEach(el => el.remove());
    
    const popup = document.createElement('div');
    popup.className = 'plutchik-popup plutchik-linkedin-popup';
    
    const top5 = Object.entries(result.all_emotions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    let adviceHTML = '';
    if (warnings.length > 0) {
      adviceHTML = `
        <div class="plutchik-advice">
          <strong>💼 Professional Tone Check:</strong>
          <ul>
            ${warnings.map(w => `<li>${w}</li>`).join('')}
          </ul>
          <p class="plutchik-advice-hint">Consider rephrasing before posting</p>
        </div>
      `;
    }
    
    popup.innerHTML = `
      <div class="plutchik-popup-header">
        <strong>${result.emotion}</strong>
        <span class="plutchik-confidence">${(result.confidence * 100).toFixed(0)}%</span>
      </div>
      ${adviceHTML}
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
   * Analyze a LinkedIn post/comment
   */
  async function analyzeElement(element) {
    if (element.classList.contains(ANALYZED_CLASS)) return;
    
    const text = extractText(element);
    if (!text || text.length < 5) return;
    
    const cacheKey = text.substring(0, 100);
    if (analysisCache.has(cacheKey)) {
      const result = analysisCache.get(cacheKey);
      element.appendChild(createToneWarning(result));
      element.classList.add(ANALYZED_CLASS);
      return;
    }
    
    chrome.runtime.sendMessage(
      { type: 'ANALYZE_TEXT', text: text },
      (response) => {
        if (response && response.success) {
          analysisCache.set(cacheKey, response);
          element.appendChild(createToneWarning(response));
        }
      }
    );
    
    element.classList.add(ANALYZED_CLASS);
  }

  /**
   * Find and analyze visible content
   */
  function analyzeVisibleContent() {
    const comments = document.querySelectorAll('.comments-comment, [class*="comment"], .feed-shared-update-v2');
    
    comments.forEach(comment => {
      if (!comment.classList.contains(ANALYZED_CLASS) && comment.offsetWidth > 100) {
        analyzeElement(comment);
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

  console.log('[Plutchik] LinkedIn overlay loaded - Professional tone advisor active');
  
  setTimeout(analyzeVisibleContent, 1500);
  
  const observer = new MutationObserver(debounce(analyzeVisibleContent, 500));
  observer.observe(document.body, { childList: true, subtree: true });
  
})();
