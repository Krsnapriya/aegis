/**
 * Plutchik ERC — Background Worker v2.2 (Privacy-First)
 * Orchestrates routing and LOCAL WASM inference via Transformers.js.
 */

import { ModelRouter } from '../models/ModelRouter.js';

// We'll use the CDN version for this scaffold to avoid heavy bundling
// In a production env, you'd bundle @xenova/transformers via Webpack/Vite
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// Skip local check for the extension environment
env.allowLocalModels = false;

let classifier = null;

/**
 * Initialize the local model
 */
async function getModel() {
  if (!classifier) {
    console.log("[Plutchik] Initializing on-device model (RoBERTa)...");
    // Using a base model placeholder — in prod, you'd point to your 'plutchik-roberta' export
    classifier = await pipeline('text-classification', 'Xenova/roberta-base', {
      device: 'webgpu' // Force WebGPU if available, fallback to WASM automatically
    });
    console.log("[Plutchik] Local model ready.");
  }
  return classifier;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'SCAN_BATCH') {
    handleBatchScanLocal(request, sendResponse);
    return true; 
  }
});

/**
 * Perform inference entirely ON-DEVICE
 */
async function handleBatchScanLocal(request, sendResponse) {
  const { items, site, modelHint } = request;
  
  // 1. Select Model via Routing Engine
  const modelType = ModelRouter.select(site, items, modelHint);
  
  try {
    const model = await getModel();
    const predictions = [];

    for (const text of items) {
      // Local WASM Inference
      const result = await model(text, { topk: 5 });
      
      // Map generic RoBERTa results to Plutchik classes
      // In a production scenario, your custom model would return Plutchik IDs directly
      predictions.push({
        emotion: result[0].label,
        confidence: result[0].score,
        sarcasm_prob: 0.1, // Local model head placeholder
        intensity: 0.5,    // Local model head placeholder
        ring: "primary",
        model_used: `local_${modelType}`
      });
    }

    sendResponse({ predictions });
    console.log(`[Plutchik] On-device inference complete for ${items.length} items.`);
    
  } catch (error) {
    console.error("[Plutchik] Local Inference Error:", error);
    // Fallback to API if local fails (optional safety valve)
    sendResponse({ predictions: [] });
  }
}
