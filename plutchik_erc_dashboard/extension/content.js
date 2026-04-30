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
  const overlay = document.createElement('div');
  overlay.className = 'plutchik-overlay';
  
  const color = getRingColor(data.ring);
  
  overlay.innerHTML = `
    <div class="plutchik-ring-indicator" style="background: ${color}"></div>
    <span class="plutchik-emotion">${data.emotion}</span>
    <span class="plutchik-confidence">${(data.confidence * 100).toFixed(0)}%</span>
    ${data.sarcasm_prob > 0.5 ? '<span class="plutchik-sarcasm">⚠️ Sarcasm</span>' : ''}
  `;
  
  // Position relative to element
  element.style.position = 'relative';
  element.appendChild(overlay);
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
