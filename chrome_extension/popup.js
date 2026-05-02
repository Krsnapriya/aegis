/**
 * Popup script for Plutchik Emotion Detector
 */

const API_URL = 'http://localhost:8000';

// DOM elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultDiv = document.getElementById('result');
const resultEmotion = document.getElementById('resultEmotion');
const resultConfidence = document.getElementById('resultConfidence');
const resultIntensity = document.getElementById('resultIntensity');
const resultSarcasm = document.getElementById('resultSarcasm');
const emotionBars = document.getElementById('emotionBars');

// Check API health on load
async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (response.ok) {
      const data = await response.json();
      statusDot.className = 'status-dot';
      statusText.textContent = `Connected (${data.device || 'local'})`;
    } else {
      throw new Error('API not responding');
    }
  } catch (error) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Offline - Start the inference server';
  }
}

// Analyze text
async function analyzeText() {
  const text = textInput.value.trim();
  if (!text) {
    alert('Please enter some text to analyze');
    return;
  }
  
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  statusDot.className = 'status-dot loading';
  
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id: 'popup' })
    });
    
    if (!response.ok) throw new Error('Analysis failed');
    
    const result = await response.json();
    displayResult(result);
    
  } catch (error) {
    alert(`Error: ${error.message}`);
    statusDot.className = 'status-dot offline';
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Emotion';
    statusDot.className = 'status-dot';
    statusText.textContent = 'Ready';
  }
}

// Display results
function displayResult(result) {
  resultDiv.style.display = 'block';
  
  // Header
  resultEmotion.textContent = result.emotion;
  resultConfidence.textContent = `${(result.confidence * 100).toFixed(0)}%`;
  
  // Stats
  resultIntensity.textContent = result.intensity;
  resultIntensity.className = `stat-value intensity-${result.intensity}`;
  
  resultSarcasm.textContent = result.sarcasm ? `⚠️ Yes (${(result.sarcasm_score * 100).toFixed(0)}%)` : 'No';
  resultSarcasm.className = result.sarcasm ? 'stat-value intensity-intense' : 'stat-value';
  
  // Emotion bars (top 5)
  const sortedEmotions = Object.entries(result.all_emotions)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  
  emotionBars.innerHTML = sortedEmotions.map(([emotion, score]) => `
    <div class="bar-row">
      <span class="bar-label">${emotion}</span>
      <div class="bar-container">
        <div class="bar" style="width: ${score * 100}%"></div>
      </div>
      <span class="bar-value">${(score * 100).toFixed(0)}%</span>
    </div>
  `).join('');
}

// Event listeners
analyzeBtn.addEventListener('click', analyzeText);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    analyzeText();
  }
});

// Initialize
checkHealth();
