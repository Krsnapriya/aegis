/**
 * Plutchik Emotion Detector - Background Service Worker
 * Handles on-device inference using WebGPU/WASM
 */

// Emotion classes (32)
const EMOTION_CLASSES = [
  'ecstasy', 'admiration', 'terror', 'amazement', 'grief', 'loathing',
  'rage', 'vigilance', 'joy', 'trust', 'fear', 'surprise', 'sadness',
  'disgust', 'anger', 'anticipation', 'serenity', 'acceptance',
  'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance',
  'interest', 'optimism', 'love', 'submission', 'awe', 'disapproval',
  'remorse', 'contempt', 'aggressiveness'
];

// Color mapping for emotion rings
const EMOTION_COLORS = {
  // Joy sector
  'joy': '#FFD700', 'ecstasy': '#FF69B4', 'serenity': '#FFE4B5',
  // Trust sector  
  'trust': '#90EE90', 'admiration': '#DA70D6', 'acceptance': '#98FB98',
  // Fear sector
  'fear': '#8B0000', 'terror': '#8B0000', 'apprehension': '#CD5C5C',
  // Surprise sector
  'surprise': '#FFA500', 'amazement': '#FF4500', 'distraction': '#FFDAB9',
  // Sadness sector
  'sadness': '#00008B', 'grief': '#2F4F4F', 'pensiveness': '#4682B4',
  // Disgust sector
  'disgust': '#006400', 'loathing': '#8B4513', 'boredom': '#556B2F',
  // Anger sector
  'anger': '#DC143C', 'rage': '#B22222', 'annoyance': '#DC143C',
  // Anticipation sector
  'anticipation': '#9370DB', 'vigilance': '#556B2F', 'interest': '#DDA0DD',
  // Dyadic emotions
  'optimism': '#FFB6C1', 'love': '#FF69B4', 'submission': '#D8BFD8',
  'awe': '#9400D3', 'disapproval': '#8B7D6B', 'remorse': '#483D8B',
  'contempt': '#8B4513', 'aggressiveness': '#B22222'
};

// Intensity ring colors
const INTENSITY_COLORS = {
  'mild': '#4A90D9',      // Blue
  'primary': '#50C878',   // Green  
  'intense': '#DC143C'    // Red
};

// Model state (lazy loaded)
let model = null;
let tokenizer = null;

/**
 * Initialize the on-device model
 * In production, this would load ONNX model via WebGPU
 */
async function initializeModel() {
  if (model) return model;
  
  console.log('[Plutchik] Loading on-device model...');
  
  // TODO: Load ONNX model from extension storage
  // For now, we use a mock that simulates the API response structure
  
  model = {
    predict: async (text, context = []) => {
      // Mock prediction - in production this runs WASM/WebGPU inference
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, context })
      }).then(r => r.json());
      
      return response;
    }
  };
  
  console.log('[Plutchik] Model loaded');
  return model;
}

/**
 * Get emotion color based on primary emotion and intensity
 */
function getEmotionIndicator(emotion, intensity) {
  const baseColor = EMOTION_COLORS[emotion] || '#808080';
  const ringColor = INTENSITY_COLORS[intensity] || INTENSITY_COLORS['primary'];
  
  return {
    dotColor: ringColor,  // Ring color indicates intensity
    hoverColor: baseColor, // Hover shows emotion-specific color
    emotion: emotion,
    intensity: intensity
  };
}

/**
 * Analyze text and return emotion data
 */
async function analyzeText(text, context = []) {
  try {
    await initializeModel();
    const result = await model.predict(text, context);
    return {
      success: true,
      ...result,
      indicator: getEmotionIndicator(result.emotion, result.intensity)
    };
  } catch (error) {
    console.error('[Plutchik] Analysis error:', error);
    return { success: false, error: error.message };
  }
}

// Message handler for content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE_TEXT') {
    analyzeText(message.text, message.context || [])
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep channel open for async response
  }
  
  if (message.type === 'GET_EMOTION_COLORS') {
    sendResponse({ colors: EMOTION_COLORS, intensityColors: INTENSITY_COLORS });
    return true;
  }
  
  if (message.type === 'MODEL_READY') {
    initializeModel().then(() => sendResponse({ ready: true }));
    return true;
  }
});

// Listen for tab updates to inject on supported sites
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Could trigger initial analysis here if needed
  }
});

console.log('[Plutchik] Background service worker initialized');
