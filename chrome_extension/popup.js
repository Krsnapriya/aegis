/**
 * Plutchik popup script – on-device mode only.
 * Analysis is delegated to the background service worker (ONNX / heuristic).
 * No direct network calls are made.
 */

// DOM elements
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const textInput  = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultDiv  = document.getElementById('result');
const resultEmotion    = document.getElementById('resultEmotion');
const resultConfidence = document.getElementById('resultConfidence');
const resultIntensity  = document.getElementById('resultIntensity');
const resultSarcasm    = document.getElementById('resultSarcasm');
const emotionBars      = document.getElementById('emotionBars');

// ─── Engine status ─────────────────────────────────────────────────────────────

function checkEngineStatus() {
  chrome.runtime.sendMessage({ action: 'get-engine-status' }, (resp) => {
    if (chrome.runtime.lastError || !resp) {
      statusDot.className  = 'status-dot offline';
      statusText.textContent = 'Extension background error';
      return;
    }
    statusDot.className  = 'status-dot';
    const modeLabel = resp.mode === 'onnx' ? 'ONNX/WASM' : 'heuristic fallback';
    statusText.textContent = resp.ready
      ? `On-device ready (${modeLabel})`
      : 'Initialising on-device engine…';
  });
}

// ─── Analysis ──────────────────────────────────────────────────────────────────

async function analyzeText() {
  const text = textInput.value.trim();
  if (!text) { alert('Please enter some text to analyse'); return; }

  analyzeBtn.disabled    = true;
  analyzeBtn.textContent = 'Analysing…';
  statusDot.className    = 'status-dot loading';
  statusText.textContent = 'Running on-device inference…';

  chrome.runtime.sendMessage({ action: 'analyzeText', text }, (resp) => {
    analyzeBtn.disabled    = false;
    analyzeBtn.textContent = 'Analyze Emotion';

    if (chrome.runtime.lastError || !resp) {
      statusDot.className    = 'status-dot offline';
      statusText.textContent = 'Error – see console';
      alert('Analysis failed: ' + (chrome.runtime.lastError?.message ?? 'Unknown error'));
      return;
    }

    if (!resp.success) {
      statusDot.className    = 'status-dot offline';
      statusText.textContent = 'Analysis error';
      alert('Analysis failed: ' + (resp.error ?? 'Unknown error'));
      return;
    }

    statusDot.className    = 'status-dot';
    const modeLabel = resp.data?.inference_mode === 'onnx' ? 'ONNX' : 'heuristic';
    statusText.textContent = `Done (${modeLabel}, ${resp.data?.processing_time_ms ?? '—'}ms)`;
    displayResult(resp.data);
  });
}

// ─── Result display ────────────────────────────────────────────────────────────

function displayResult(data) {
  resultDiv.style.display = 'block';

  const topEmotion = data.emotions?.[0];
  resultEmotion.textContent    = topEmotion?.emotion ?? data.primary_emotion ?? '—';
  resultConfidence.textContent = topEmotion ? `${(topEmotion.confidence * 100).toFixed(0)}%` : '—';

  resultIntensity.textContent  = data.intensity ?? '—';
  resultIntensity.className    = `stat-value intensity-${data.intensity ?? 'primary'}`;

  const sarcasmPct = ((data.sarcasm_probability ?? 0) * 100).toFixed(0);
  const isSarcastic = (data.sarcasm_probability ?? 0) > 0.3;
  resultSarcasm.textContent = isSarcastic ? `⚠️ ${sarcasmPct}%` : `No (${sarcasmPct}%)`;
  resultSarcasm.className   = isSarcastic ? 'stat-value intensity-intense' : 'stat-value';

  // Top 5 emotion bars
  const sorted = (data.emotions ?? []).slice(0, 5);
  emotionBars.innerHTML = sorted.map(({ emotion, confidence }) => `
    <div class="bar-row">
      <span class="bar-label">${emotion}</span>
      <div class="bar-container">
        <div class="bar" style="width: ${(confidence * 100).toFixed(1)}%"></div>
      </div>
      <span class="bar-value">${(confidence * 100).toFixed(0)}%</span>
    </div>
  `).join('');
}

// ─── Event listeners ───────────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', analyzeText);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) analyzeText();
});

// Init
checkEngineStatus();
