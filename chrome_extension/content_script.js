// Plutchik Dynamic Coach – On-Demand Content Script
// Activates only when the user explicitly requests analysis.
// Privacy guarantee: analysis is delegated to the background service worker,
// which runs fully on-device via ONNX/WebAssembly. Zero network requests.

let isActive = false;
let currentAnalysisPanel = null;

// ─── Unique request ID helper ──────────────────────────────────────────────────
function generateRequestId() {
  const ts  = Date.now();
  const rnd = (typeof crypto !== 'undefined' && crypto.randomUUID)
    ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
    : Math.random().toString(36).slice(2, 14);
  return `req_${ts}_${rnd}`;
}

// ─── Activation ────────────────────────────────────────────────────────────────

function initializeOnDemand() {
  if (isActive) return;
  isActive = true;
  console.log('[Plutchik] Coach activated');

  addActivationIndicator();
  scanAndInjectButtons();
  document.addEventListener('mouseup', handleTextSelection);
  observeDynamicContent();
}

function addActivationIndicator() {
  if (document.getElementById('plutchik-activation-indicator')) return;

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
    " id="plutchik-indicator-inner">
      <span>✨</span>
      <span>Plutchik Active</span>
      <span style="font-size: 16px;">×</span>
    </div>
  `;
  document.body.appendChild(indicator);

  document.getElementById('plutchik-indicator-inner').addEventListener('click', () => {
    indicator.remove();
    plutchikDeactivate();
  });
}

function plutchikDeactivate() {
  isActive = false;
  if (currentAnalysisPanel) { currentAnalysisPanel.remove(); currentAnalysisPanel = null; }
  console.log('[Plutchik] Coach deactivated');
}

// ─── Button injection ──────────────────────────────────────────────────────────

function scanAndInjectButtons() {
  const targets = document.querySelectorAll(
    'textarea, [contenteditable="true"], input[type="text"]'
  );
  targets.forEach(el => {
    if (el.dataset.plutchikInjected) return;
    el.dataset.plutchikInjected = 'true';

    const btn = document.createElement('button');
    btn.className = 'plutchik-analyze-btn';
    btn.innerHTML = '🔍 Analyze Tone';
    btn.style.cssText = `
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

    el.addEventListener('focus', () => {
      btn.style.opacity = '1';
      btn.style.transform = 'translateY(0)';
    });
    el.addEventListener('blur', () => {
      btn.style.opacity = '0';
      btn.style.transform = 'translateY(5px)';
    });
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const text = el.value || el.innerText;
      if (text.trim()) analyzeText(text, el);
    });

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'position: relative; display: inline-block; width: 100%;';
    el.parentNode.insertBefore(wrapper, el);
    wrapper.appendChild(el);
    wrapper.appendChild(btn);
  });
}

// ─── Text selection tooltip ────────────────────────────────────────────────────

function handleTextSelection(e) {
  const selected = window.getSelection().toString().trim();
  if (selected.length > 0 && selected.length < 1000) {
    showSelectionTooltip(selected, e);
  }
}

function showSelectionTooltip(text, event) {
  const existing = document.getElementById('plutchik-selection-tooltip');
  if (existing) existing.remove();

  const tooltip = document.createElement('div');
  tooltip.id = 'plutchik-selection-tooltip';
  const inner = document.createElement('div');
  inner.style.cssText = `
    position: fixed;
    left: ${event.clientX + 10}px;
    top: ${event.clientY - 40}px;
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
  `;
  inner.textContent = '🔍 Analyze with Plutchik';
  inner.addEventListener('click', () => { analyzeText(text, null, event.target); tooltip.remove(); });
  tooltip.appendChild(inner);
  document.body.appendChild(tooltip);

  // Auto-hide after 3 seconds
  setTimeout(() => tooltip.remove(), 3000);
}

// ─── Dynamic content observer ──────────────────────────────────────────────────

function observeDynamicContent() {
  const observer = new MutationObserver((mutations) => {
    if (mutations.some(m => m.addedNodes.length > 0)) {
      setTimeout(scanAndInjectButtons, 500);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

// ─── Core analysis function ────────────────────────────────────────────────────

async function analyzeText(text, targetElement, referenceElement = null) {
  const requestId = generateRequestId();
  showLoadingPanel(targetElement, referenceElement);

  try {
    // Delegate to background service worker for on-device inference (privacy-first).
    const response = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { action: 'analyzeText', text, requestId },
        (resp) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(resp);
          }
        }
      );
    });

    if (response && response.success) {
      displayAnalysis(response.data, targetElement);
    } else {
      throw new Error(response?.error || 'Analysis failed');
    }
  } catch (error) {
    console.error('[Plutchik] Analysis error:', error);
    showErrorPanel(error.message);
  }
}

// ─── Loading panel ─────────────────────────────────────────────────────────────

function showLoadingPanel(targetElement, referenceElement) {
  if (currentAnalysisPanel) currentAnalysisPanel.remove();

  const panel = document.createElement('div');
  panel.id = 'plutchik-analysis-panel';
  panel.style.cssText = 'position: fixed; z-index: 999998;';
  panel.innerHTML = `
    <div style="
      background: white;
      border: 2px solid #667eea;
      border-radius: 12px;
      padding: 20px;
      max-width: 400px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
      animation: plutchikFadeIn 0.3s ease;
    ">
      <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <div style="
          width: 20px; height: 20px;
          border: 3px solid #667eea;
          border-top-color: transparent;
          border-radius: 50%;
          animation: plutchikSpin 1s linear infinite;
        "></div>
        <h3 style="margin: 0; color: #667eea; font-size: 16px;">Analysing…</h3>
      </div>
      <p style="margin: 0; color: #666; font-size: 14px;">
        Checking emotional tone, sarcasm, and trajectory — on your device 🔒
      </p>
    </div>
  `;
  document.body.appendChild(panel);
  currentAnalysisPanel = panel;
  positionPanel(panel, targetElement, referenceElement);
}

// ─── Analysis results panel ────────────────────────────────────────────────────

function displayAnalysis(data, targetElement) {
  if (!currentAnalysisPanel) return;

  const riskLevel   = data.risk_level || 'low';
  const sarcasmProb = data.sarcasm_probability || 0;
  const topEmotion  = data.emotions?.[0]?.emotion || data.primary_emotion || 'Unknown';
  const intensity   = data.intensity || 'primary';
  const reframes    = data.reframe_suggestions || [];
  const mode        = data.inference_mode || 'heuristic';

  const riskColor = riskLevel === 'high' ? '#ef4444' : riskLevel === 'medium' ? '#f59e0b' : '#10b981';
  const riskLabel = riskLevel === 'high' ? '⚠️ High Risk' : riskLevel === 'medium' ? '🟡 Medium Risk' : '✅ Low Risk';

  const privacyBadge = `
    <div style="font-size: 10px; color: #6b7280; text-align: right; margin-bottom: 8px;">
      🔒 On-device · ${mode === 'onnx' ? 'ONNX/WASM' : 'heuristic'} · ${data.processing_time_ms ?? '—'}ms
    </div>`;

  const sarcasmSection = `
    <div style="
      margin-bottom: 12px; padding: 10px;
      background: ${sarcasmProb > 0.3 ? '#fef3c7' : '#f0fdf4'};
      border-radius: 8px;
      border-left: 4px solid ${sarcasmProb > 0.3 ? '#f59e0b' : '#10b981'};
    ">
      <div style="font-size: 11px; color: #64748b; margin-bottom: 3px;">Sarcasm / Incongruity</div>
      <div style="font-size: 16px; font-weight: 700; color: ${sarcasmProb > 0.3 ? '#92400e' : '#065f46'};">
        ${(sarcasmProb * 100).toFixed(0)}% ${sarcasmProb > 0.3 ? '😏 Likely Sarcastic' : '😊 Sincere'}
      </div>
      ${data.sarcasm_signals?.length ? `<div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">${data.sarcasm_signals[0]}</div>` : ''}
    </div>`;

  const trajectorySection = (data.trajectory_forecast?.length > 0) ? `
    <div style="margin-bottom: 12px; padding: 10px; background: #eff6ff; border-radius: 8px;">
      <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">📈 Trajectory Forecast</div>
      <div style="font-size: 13px; color: #1e40af; line-height: 1.5;">
        ${getTrajectorySummary(data.trajectory_forecast)}
      </div>
    </div>` : '';

  const reframesSection = reframes.length > 0 ? `
    <div style="margin-bottom: 12px;">
      <div style="font-size: 11px; color: #64748b; margin-bottom: 6px; font-weight: 600;">💡 Suggested Reframes (click to replace):</div>
      ${reframes.slice(0, 3).map(r => `
        <div
          style="
            padding: 8px 10px; margin-bottom: 6px;
            background: white; border: 1px solid #e2e8f0;
            border-radius: 6px; font-size: 13px; color: #334155;
            cursor: pointer; transition: all 0.15s;
          "
          onmouseover="this.style.background='#f8fafc'; this.style.borderColor='#667eea'"
          onmouseout="this.style.background='white'; this.style.borderColor='#e2e8f0'"
          onclick="(function(el, t) {
            var active = document.activeElement;
            if (active && (active.tagName === 'TEXTAREA' || active.isContentEditable)) {
              if (active.tagName === 'TEXTAREA') active.value = t;
              else active.innerText = t;
            } else {
              navigator.clipboard.writeText(t);
            }
            el.style.borderColor='#10b981'; el.style.background='#f0fdf4';
          })(this, ${JSON.stringify(r)})"
        >
          ${r}
          <div style="font-size: 10px; color: #667eea; margin-top: 3px;">Click to replace / copy</div>
        </div>
      `).join('')}
    </div>` : '';

  currentAnalysisPanel.innerHTML = `
    <div style="
      position: fixed;
      background: white;
      border: 2px solid ${riskColor};
      border-radius: 12px;
      padding: 16px;
      max-width: 440px;
      z-index: 999998;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
      animation: plutchikFadeIn 0.3s ease;
    " id="plutchik-panel-inner">
      ${privacyBadge}
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <h3 style="margin: 0; color: ${riskColor}; font-size: 17px; font-weight: 700;">${riskLabel}</h3>
        <button
          onclick="document.getElementById('plutchik-analysis-panel').remove()"
          style="background: none; border: none; font-size: 20px; cursor: pointer; color: #999; line-height: 1;"
        >×</button>
      </div>
      <div style="margin-bottom: 12px; padding: 10px; background: #f8fafc; border-radius: 8px;">
        <div style="font-size: 11px; color: #64748b; margin-bottom: 3px;">Primary Emotion</div>
        <div style="font-size: 19px; font-weight: 700; color: #1e293b; text-transform: capitalize;">${topEmotion}</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 3px; text-transform: capitalize;">Intensity: ${intensity}</div>
      </div>
      ${sarcasmSection}
      ${trajectorySection}
      ${reframesSection}
    </div>
  `;

  // Position the inner panel (it has position:fixed itself)
  positionPanel(currentAnalysisPanel.querySelector('#plutchik-panel-inner'), targetElement, null);
}

function getTrajectorySummary(forecast) {
  if (!forecast || forecast.length < 2) return 'No trajectory data';
  // Compare dominant emotion index at step 0 vs step 9
  const topAt = step => step.indexOf(Math.max(...step));
  const start = topAt(forecast[0]);
  const end   = topAt(forecast[forecast.length - 1]);
  if (end > start) return 'Conversation may escalate emotionally';
  if (end < start) return 'Conversation likely to calm down';
  return 'Emotional tone expected to remain stable';
}

function showErrorPanel(errorMessage) {
  if (!currentAnalysisPanel) return;
  currentAnalysisPanel.innerHTML = `
    <div style="
      position: fixed;
      background: white;
      border: 2px solid #ef4444;
      border-radius: 12px;
      padding: 20px;
      max-width: 400px;
      z-index: 999998;
      box-shadow: 0 8px 30px rgba(0,0,0,0.2);
      top: 80px; right: 20px;
    ">
      <h3 style="margin: 0 0 10px; color: #ef4444; font-size: 16px;">❌ Analysis Failed</h3>
      <p style="margin: 0; color: #666; font-size: 14px;">${errorMessage}</p>
      <p style="margin: 10px 0 0; color: #999; font-size: 12px;">
        Ensure the extension is installed correctly. On-device inference requires no server.
      </p>
      <button onclick="document.getElementById('plutchik-analysis-panel').remove()" style="
        margin-top: 14px; background: #ef4444; color: white;
        border: none; padding: 8px 16px; border-radius: 6px;
        cursor: pointer; font-size: 13px;
      ">Close</button>
    </div>
  `;
}

function positionPanel(panel, targetElement, referenceElement) {
  if (!panel) return;
  const rect = referenceElement?.getBoundingClientRect()
    || targetElement?.getBoundingClientRect()
    || null;
  if (rect) {
    panel.style.left = `${Math.min(rect.right + 10, window.innerWidth - 460)}px`;
    panel.style.top  = `${Math.max(10, Math.min(rect.top, window.innerHeight - 500))}px`;
  } else {
    panel.style.right = '20px';
    panel.style.top   = '80px';
  }
}

// ─── Message listener (from background) ───────────────────────────────────────

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
  } else if (request.action === 'trigger-selection-analysis') {
    const selected = window.getSelection().toString().trim();
    if (isActive && selected) analyzeText(selected, null);
  } else if (request.action === 'show-activation-prompt') {
    if (!isActive) {
      const note = document.createElement('div');
      note.style.cssText = `
        position: fixed; bottom: 70px; right: 20px;
        background: #1e293b; color: white;
        padding: 10px 16px; border-radius: 8px;
        font-size: 13px; z-index: 999999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      `;
      note.textContent = 'Press Ctrl+Shift+P to activate Plutchik first';
      document.body.appendChild(note);
      setTimeout(() => note.remove(), 3000);
    }
  }
});

console.log('[Plutchik] Content script loaded. Press Ctrl+Shift+P or click extension icon to activate.');
