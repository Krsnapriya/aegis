/**
 * Plutchik Emotion Lens — Content Script
 */

console.log('Plutchik Emotion Lens active');

// Debounce to prevent API spam
let analysisTimeout;

document.addEventListener('mouseover', (e) => {
  const target = e.target;
  
  // Identify relevant elements (Reddit comments, LinkedIn posts, etc.)
  const isPostOrComment = 
    target.closest('.comment') || 
    target.closest('[data-testid="comment"]') ||
    target.closest('.feed-shared-update-v2__description') ||
    target.closest('.ii.gt'); // Gmail message body

  if (isPostOrComment && target.innerText.length > 10) {
    clearTimeout(analysisTimeout);
    analysisTimeout = setTimeout(() => {
      analyzeElement(target.closest('.comment') || target.closest('[data-testid="comment"]') || target);
    }, 1000);
  }
});

async function analyzeElement(element) {
  if (element.dataset.plutchikAnalyzed) return;
  
  const text = element.innerText.slice(0, 500); // Limit length
  
  chrome.runtime.sendMessage({ type: 'PREDICT_EMOTION', text }, (response) => {
    if (response && response.success) {
      injectOverlay(element, response.data);
      element.dataset.plutchikAnalyzed = 'true';
    }
  });
}

function injectOverlay(element, data) {
  const host = document.createElement('div');
  host.className = 'plutchik-host';
  const shadow = host.attachShadow({ mode: 'closed' });
  const color = getRingColor(data.ring);
  shadow.innerHTML = `
    <style>
      :host { all: initial; font-family: system-ui, -apple-system, sans-serif; }
      .plutchik-overlay {
        position: absolute; top: 4px; right: 4px; z-index: 2147483646;
        display: flex; align-items: center; gap: 6px; padding: 4px 8px;
        border-radius: 8px; background: rgba(13,17,23,0.92); color: #e6edf3;
        font-size: 11px; border: 1px solid rgba(48,54,61,0.9); pointer-events: none;
      }
      .plutchik-ring-indicator { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    </style>
    <div class="plutchik-overlay">
      <div class="plutchik-ring-indicator" style="background: ${color}"></div>
      <span>${data.emotion}</span>
      <span>${(data.confidence * 100).toFixed(0)}%</span>
      ${data.sarcasm_prob > 0.5 ? '<span>⚠ sarcasm</span>' : ''}
    </div>
  `;
  element.style.position = 'relative';
  element.appendChild(host);
}

function getRingColor(ring) {
  const colors = {
    intense: '#ff7b72',
    primary: '#58a6ff',
    mild: '#a371f7',
    dyadic: '#3fb950'
  };
  return colors[ring] || '#8b949e';
}
