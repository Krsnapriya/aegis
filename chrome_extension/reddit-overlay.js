/**
 * Reddit Overlay - Plutchik Emotion Detector
 * Adds emotion indicators to Reddit comments
 */

(function() {
  'use strict';

  // Cache for analyzed comments
  const analysisCache = new Map();
  const ANALYZED_CLASS = 'plutchik-analyzed';

  /**
   * Extract comment text from Reddit comment element
   */
  function extractCommentText(commentEl) {
    // Reddit uses data-testid="comment" for comments
    const textEl = commentEl.querySelector('[data-testid="comment"]') || 
                   commentEl.querySelector('.md') ||
                   commentEl.querySelector('p');
    return textEl ? textEl.textContent.trim() : '';
  }

  /**
   * Create emotion indicator dot
   */
  function createEmotionIndicator(result) {
    const container = document.createElement('div');
    container.className = 'plutchik-indicator';
    container.title = `${result.emotion} (${result.confidence.toFixed(1)}) - ${result.intensity}\nSarcasm: ${result.sarcasm ? '⚠️ Yes' : 'No'}`;
    
    // Create the colored dot
    const dot = document.createElement('span');
    dot.className = 'plutchik-dot';
    dot.style.backgroundColor = result.indicator.dotColor;
    
    // Add sarcasm flag if detected
    if (result.sarcasm && result.sarcasm_score > 0.7) {
      dot.classList.add('plutchik-sarcastic');
      dot.title += '\n⚠️ High sarcasm detected!';
    }
    
    container.appendChild(dot);
    
    // Click to show full details
    container.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      showEmotionDetails(container, result);
    });
    
    return container;
  }

  /**
   * Show detailed emotion breakdown in a tooltip/popup
   */
  function showEmotionDetails(anchorEl, result) {
    // Remove existing popups
    document.querySelectorAll('.plutchik-popup').forEach(el => el.remove());
    
    const popup = document.createElement('div');
    popup.className = 'plutchik-popup';
    
    // Build HTML content
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
                <div class="plutchik-bar" style="width: ${score * 100}%; background-color: getEmotionColor('${emotion}')"></div>
              </div>
              <span class="plutchik-bar-value">${(score * 100).toFixed(0)}%</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
    
    // Position popup
    anchorEl.appendChild(popup);
    
    // Close on outside click
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
   * Analyze a single comment and add indicator
   */
  async function analyzeComment(commentEl) {
    if (commentEl.classList.contains(ANALYZED_CLASS)) return;
    
    const text = extractCommentText(commentEl);
    if (!text || text.length < 5) return;
    
    // Check cache first
    const cacheKey = text.substring(0, 100);
    if (analysisCache.has(cacheKey)) {
      const result = analysisCache.get(cacheKey);
      commentEl.appendChild(createEmotionIndicator(result));
      commentEl.classList.add(ANALYZED_CLASS);
      return;
    }
    
    // Request analysis from background script
    chrome.runtime.sendMessage(
      { type: 'ANALYZE_TEXT', text: text },
      (response) => {
        if (response && response.success) {
          analysisCache.set(cacheKey, response);
          commentEl.appendChild(createEmotionIndicator(response));
        }
      }
    );
    
    commentEl.classList.add(ANALYZED_CLASS);
  }

  /**
   * Find and analyze all visible comments
   */
  function analyzeVisibleComments() {
    // Reddit comment selector
    const comments = document.querySelectorAll('[data-testid="comment"], .Comment, .comment');
    
    comments.forEach(comment => {
      if (!comment.classList.contains(ANALYZED_CLASS)) {
        analyzeComment(comment);
      }
    });
  }

  /**
   * Debounce function for scroll events
   */
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Initialize
  console.log('[Plutchik] Reddit overlay loaded');
  
  // Initial analysis
  setTimeout(analyzeVisibleComments, 1000);
  
  // Observe DOM changes for lazy-loaded comments
  const observer = new MutationObserver(debounce(analyzeVisibleComments, 500));
  observer.observe(document.body, { childList: true, subtree: true });
  
  // Also observe on scroll (for infinite scroll)
  window.addEventListener('scroll', debounce(analyzeVisibleComments, 500));

})();
