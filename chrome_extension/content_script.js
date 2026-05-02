// Plutchik Dynamic Coach - On-Demand Content Script
// Only activates when user explicitly requests analysis

let isActive = false;
let currentAnalysisPanel = null;
let requestIdCounter = 0;

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Initialize only when user activates
function initializeOnDemand() {
  if (isActive) return;
  
  isActive = true;
  console.log('[Plutchik] Coach activated');
  
  // Add activation indicator
  addActivationIndicator();
  
  // Scan for text areas and add analyze buttons
  scanAndInjectButtons();
  
  // Listen for text selection
  document.addEventListener('mouseup', handleTextSelection);
  
  // Listen for dynamic content (new comments/replies)
  observeDynamicContent();
}

// Add a small floating button to activate/deactivate
function addActivationIndicator() {
  const indicator = document.createElement('div');
  indicator.id = 'plutchik-activation-indicator';
  indicator.innerHTML = `
    <div style="
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 12px 16px;
      border-radius: 50px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      z-index: 999999;
      box-shadow: 0 4px 15px rgba(0,0,0,0.3);
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    " onclick="this.parentElement.remove(); window.plutchikDeactivate();">
      <span>✨</span>
      <span>Plutchik Active</span>
      <span style="font-size: 16px;">×</span>
    </div>
  `;
  document.body.appendChild(indicator);
}

// Deactivation function exposed to global scope
window.plutchikDeactivate = () => {
  isActive = false;
  if (currentAnalysisPanel) {
    currentAnalysisPanel.remove();
    currentAnalysisPanel = null;
  }
  console.log('[Plutchik] Coach deactivated');
};

// Scan page for text areas and inject analyze buttons
function scanAndInjectButtons() {
  const textAreas = document.querySelectorAll('textarea, [contenteditable="true"], input[type="text"]');
  
  textAreas.forEach((textarea, index) => {
    if (textarea.dataset.plutchikInjected) return;
    
    textarea.dataset.plutchikInjected = 'true';
    
    // Create analyze button
    const button = document.createElement('button');
    button.className = 'plutchik-analyze-btn';
    button.innerHTML = '🔍 Analyze Tone';
    button.style.cssText = `
      position: absolute;
      right: 8px;
      bottom: 8px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      z-index: 1000;
      opacity: 0;
      transform: translateY(5px);
      transition: all 0.2s ease;
    `;
    
    // Show button on focus
    textarea.addEventListener('focus', () => {
      button.style.opacity = '1';
      button.style.transform = 'translateY(0)';
    });
    
    textarea.addEventListener('blur', () => {
      button.style.opacity = '0';
      button.style.transform = 'translateY(5px)';
    });
    
    // Analyze on click
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const text = textarea.value || textarea.innerText;
      if (text.trim()) {
        analyzeText(text, textarea);
      }
    });
    
    // Position button relative to textarea
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    wrapper.style.width = '100%';
    
    textarea.parentNode.insertBefore(wrapper, textarea);
    wrapper.appendChild(textarea);
    wrapper.appendChild(button);
  });
}

// Handle text selection for context menu analysis
function handleTextSelection(e) {
  const selection = window.getSelection();
  const selectedText = selection.toString().trim();
  
  if (selectedText.length > 0 && selectedText.length < 1000) {
    // Show quick analyze tooltip
    showSelectionTooltip(selectedText, e);
  }
}

function showSelectionTooltip(text, event) {
  // Remove existing tooltip
  const existing = document.getElementById('plutchik-selection-tooltip');
  if (existing) existing.remove();
  
  const tooltip = document.createElement('div');
  tooltip.id = 'plutchik-selection-tooltip';
  tooltip.innerHTML = `
    <div style="
      position: absolute;
      background: white;
      border: 2px solid #667eea;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      color: #667eea;
      cursor: pointer;
      z-index: 999999;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      white-space: nowrap;
    ">
      🔍 Analyze with Plutchik
    </div>
  `;
  
  const btn = tooltip.querySelector('div');
  btn.addEventListener('click', () => {
    analyzeText(text, null, event.target);
    tooltip.remove();
  });
  
  document.body.appendChild(tooltip);
  
  // Position near selection
  const rect = event.target.getBoundingClientRect();
  tooltip.querySelector('div').style.position = 'fixed';
  tooltip.querySelector('div').style.left = `${event.clientX + 10}px`;
  tooltip.querySelector('div').style.top = `${event.clientY - 40}px`;
}

// Observe dynamically added content (new comments, replies)
function observeDynamicContent() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.addedNodes.length > 0) {
        setTimeout(() => scanAndInjectButtons(), 500);
      }
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Main analysis function
async function analyzeText(text, targetElement, referenceElement = null) {
  const requestId = `req_${Date.now()}_${++requestIdCounter}`;
  
  // Show loading state
  showLoadingPanel(targetElement, referenceElement);
  
  try {
    const response = await fetch(`${API_BASE_URL}/analyze/dynamic`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        session_id: requestId,
        include_trajectory: true,
        include_reframes: true
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    displayAnalysis(data, targetElement);
    
  } catch (error) {
    console.error('[Plutchik] Analysis failed:', error);
    showErrorPanel(error.message);
  }
}

function showLoadingPanel(targetElement, referenceElement) {
  // Remove existing panel
  if (currentAnalysisPanel) {
    currentAnalysisPanel.remove();
  }
  
  const panel = document.createElement('div');
  panel.id = 'plutchik-analysis-panel';
  panel.innerHTML = `
    <div style="
      background: white;
      border: 2px solid #667eea;
      border-radius: 12px;
      padding: 20px;
      max-width: 400px;
      z-index: 999998;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
      animation: plutchikFadeIn 0.3s ease;
    ">
      <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <div style="
          width: 20px;
          height: 20px;
          border: 3px solid #667eea;
          border-top-color: transparent;
          border-radius: 50%;
          animation: plutchikSpin 1s linear infinite;
        "></div>
        <h3 style="margin: 0; color: #667eea; font-size: 16px;">Analyzing...</h3>
      </div>
      <p style="margin: 0; color: #666; font-size: 14px;">Checking emotional tone, sarcasm, and trajectory</p>
    </div>
  `;
  
  document.body.appendChild(panel);
  currentAnalysisPanel = panel;
  
  // Position panel
  positionPanel(panel, targetElement, referenceElement);
}

function displayAnalysis(data, targetElement) {
  if (!currentAnalysisPanel) return;
  
  const riskLevel = data.risk_level || 'low';
  const sarcasmProb = data.sarcasm_probability || 0;
  const primaryEmotion = data.emotions?.[0]?.emotion || 'Unknown';
  const intensity = data.intensity || 'medium';
  const reframeSuggestions = data.reframe_suggestions || [];
  
  let riskColor = '#10b981'; // green
  let riskLabel = 'Low Risk';
  if (riskLevel === 'medium') {
    riskColor = '#f59e0b'; // orange
    riskLabel = 'Medium Risk';
  } else if (riskLevel === 'high') {
    riskColor = '#ef4444'; // red
    riskLabel = 'High Risk';
  }
  
  let html = `
    <div style="
      background: white;
      border: 2px solid ${riskColor};
      border-radius: 12px;
      padding: 20px;
      max-width: 450px;
      z-index: 999998;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
      animation: plutchikFadeIn 0.3s ease;
    ">
      <!-- Header -->
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h3 style="margin: 0; color: ${riskColor}; font-size: 18px; font-weight: 700;">
          ${riskLevel === 'high' ? '⚠️' : '✅'} ${riskLabel}
        </h3>
        <button onclick="document.getElementById('plutchik-analysis-panel').remove()" style="
          background: none;
          border: none;
          font-size: 20px;
          cursor: pointer;
          color: #999;
        ">×</button>
      </div>
      
      <!-- Primary Emotion -->
      <div style="margin-bottom: 15px; padding: 12px; background: #f8fafc; border-radius: 8px;">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Primary Emotion</div>
        <div style="font-size: 20px; font-weight: 700; color: #1e293b;">${primaryEmotion}</div>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Intensity: ${intensity}</div>
      </div>
      
      <!-- Sarcasm Detection -->
      <div style="margin-bottom: 15px; padding: 12px; background: ${sarcasmProb > 0.3 ? '#fef3c7' : '#f0fdf4'}; border-radius: 8px; border-left: 4px solid ${sarcasmProb > 0.3 ? '#f59e0b' : '#10b981'};">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Sarcasm Probability</div>
        <div style="font-size: 18px; font-weight: 700; color: ${sarcasmProb > 0.3 ? '#92400e' : '#065f46'};">
          ${(sarcasmProb * 100).toFixed(0)}% ${sarcasmProb > 0.3 ? '😏 Likely Sarcastic' : '😊 Sincere'}
        </div>
      </div>
      
      <!-- Trajectory Forecast -->
      ${data.trajectory_forecast ? `
        <div style="margin-bottom: 15px; padding: 12px; background: #eff6ff; border-radius: 8px;">
          <div style="font-size: 12px; color: #64748b; margin-bottom: 6px;">📈 Predicted Trajectory</div>
          <div style="font-size: 13px; color: #1e40af; line-height: 1.5;">
            ${getTrajectorySummary(data.trajectory_forecast)}
          </div>
        </div>
      ` : ''}
      
      <!-- Reframe Suggestions -->
      ${reframeSuggestions.length > 0 ? `
        <div style="margin-bottom: 15px;">
          <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600;">💡 Suggested Reframes:</div>
          ${reframeSuggestions.slice(0, 2).map(suggestion => `
            <div style="
              padding: 10px;
              margin-bottom: 8px;
              background: white;
              border: 1px solid #e2e8f0;
              border-radius: 6px;
              font-size: 13px;
              color: #334155;
              cursor: pointer;
              transition: all 0.2s;
            " onmouseover="this.style.background='#f8fafc'; this.style.borderColor='#667eea'" 
               onmouseout="this.style.background='white'; this.style.borderColor='#e2e8f0'"
               onclick="navigator.clipboard.writeText('${suggestion.replace(/'/g, "\\'")}'); alert('Copied to clipboard!')">
              ${suggestion}
              <div style="font-size: 11px; color: #667eea; margin-top: 4px;">Click to copy</div>
            </div>
          `).join('')}
        </div>
      ` : ''}
      
      <!-- Baseline Deviation -->
      ${data.baseline_deviation ? `
        <div style="padding: 10px; background: #fef2f2; border-radius: 6px; border-left: 3px solid #ef4444;">
          <div style="font-size: 12px; color: #991b1b;">
            <strong>⚠️ Unusual for you:</strong> ${data.baseline_deviation.message}
          </div>
        </div>
      ` : ''}
    </div>
  `;
  
  currentAnalysisPanel.innerHTML = html;
}

function getTrajectorySummary(forecast) {
  if (!forecast || forecast.length === 0) return 'No trajectory data available';
  
  // Simple summary based on first few predictions
  const emotions = forecast.slice(0, 3).map(step => {
    const maxIdx = step.indexOf(Math.max(...step));
    return maxIdx;
  });
  
  const isEscalating = emotions.every((v, i, arr) => i === 0 || v >= arr[i-1]);
  const isDeescalating = emotions.every((v, i, arr) => i === 0 || v <= arr[i-1]);
  
  if (isEscalating) return 'Conversation may escalate emotionally in next messages';
  if (isDeescalating) return 'Conversation likely to calm down';
  return 'Emotional tone expected to remain stable';
}

function showErrorPanel(errorMessage) {
  if (!currentAnalysisPanel) return;
  
  currentAnalysisPanel.innerHTML = `
    <div style="
      background: white;
      border: 2px solid #ef4444;
      border-radius: 12px;
      padding: 20px;
      max-width: 400px;
      z-index: 999998;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    ">
      <h3 style="margin: 0 0 10px 0; color: #ef4444; font-size: 16px;">❌ Analysis Failed</h3>
      <p style="margin: 0; color: #666; font-size: 14px;">${errorMessage}</p>
      <p style="margin: 10px 0 0 0; color: #999; font-size: 12px;">Make sure the Plutchik server is running on localhost:8000</p>
      <button onclick="document.getElementById('plutchik-analysis-panel').remove()" style="
        margin-top: 15px;
        background: #ef4444;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
      ">Close</button>
    </div>
  `;
}

function positionPanel(panel, targetElement, referenceElement) {
  const rect = referenceElement?.getBoundingClientRect() || 
               targetElement?.getBoundingClientRect() || 
               { top: 100, left: 100 };
  
  panel.style.position = 'fixed';
  panel.style.left = `${Math.min(rect.right + 10, window.innerWidth - 460)}px`;
  panel.style.top = `${Math.min(rect.top, window.innerHeight - 400)}px`;
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'activate') {
    initializeOnDemand();
    sendResponse({ status: 'activated' });
  } else if (request.action === 'analyze') {
    if (isActive && request.text) {
      analyzeText(request.text, null);
      sendResponse({ status: 'analyzing' });
    } else {
      sendResponse({ status: 'not_active' });
    }
  }
});

console.log('[Plutchik] Content script loaded. Press Ctrl+Shift+P or click extension icon to activate.');
