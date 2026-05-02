/**
 * Plutchik On-Device Inference Engine
 * Privacy-First: All processing happens in the browser using WebGPU/WASM
 * Zero data leaves the device
 */

import * as ort from 'onnxruntime-web';

class PlutchikOnDeviceInference {
  constructor() {
    this.session = null;
    this.tokenizer = null;
    this.config = null;
    this.initialized = false;
  }

  async initialize() {
    console.log('[Plutchik] Initializing on-device inference...');
    
    // Load model configuration
    const configResponse = await fetch(chrome.runtime.getURL('model/config.json'));
    this.config = await configResponse.json();
    
    // Initialize tokenizer (simple wordpiece-like tokenizer)
    this.tokenizer = await this.loadTokenizer();
    
    // Load ONNX model sessions - pipeline: encoder -> pooled_output -> heads
    
    // Step 1: Load encoder (BERT) that produces pooled output
    this.encoderSession = await ort.InferenceSession.create(
      chrome.runtime.getURL('model/encoder.onnx'),
      {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all'
      }
    );
    
    // Step 2: Load emotion head (takes pooled_output as input)
    this.emotionSession = await ort.InferenceSession.create(
      chrome.runtime.getURL('model/emotion_head.onnx'),
      {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all'
      }
    );
    
    // Step 3: Load sarcasm head
    this.sarcasmSession = await ort.InferenceSession.create(
      chrome.runtime.getURL('model/sarcasm_head.onnx'),
      {
        executionProviders: ['webgpu', 'wasm']
      }
    );
    
    // Step 4: Load intensity head
    this.intensitySession = await ort.InferenceSession.create(
      chrome.runtime.getURL('model/intensity_head.onnx'),
      {
        executionProviders: ['webgpu', 'wasm']
      }
    );
    
    this.initialized = true;
    console.log('[Plutchik] ✓ On-device inference ready');
  }

  async loadTokenizer() {
    // Simple tokenizer - in production, use DistilBERT tokenizer
    return {
      encode: (text) => {
        const tokens = text.toLowerCase().split(/\s+/).slice(0, 128);
        const ids = tokens.map(t => this.hashToken(t) % 1000);
        // Pad to max length
        while (ids.length < 128) ids.push(0);
        return new BigInt64Array(ids);
      }
    };
  }

  hashToken(token) {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      const char = token.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  async analyze(text) {
    if (!this.initialized) {
      throw new Error('Plutchik not initialized. Call initialize() first.');
    }

    const startTime = performance.now();

    // Tokenize input - convert text to BERT input IDs
    const tokens = this.tokenizer.encode(text);
    const inputIds = new Float32Array(tokens);
    const attentionMask = new Float32Array(tokens.map(id => id !== 0 ? 1 : 0));

    // Step 1: Run encoder to get pooled output (CLS token)
    const encoderInputs = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(tokens), [1, tokens.length]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask), [1, attentionMask.length])
    };
    
    const encoderResults = await this.encoderSession.run(encoderInputs);
    const pooledOutput = encoderResults.pooled_output;

    // Step 2: Run emotion head
    const emotionResults = await this.emotionSession.run({
      pooled_output: pooledOutput
    });
    
    // Step 3: Run sarcasm head
    const sarcasmResults = await this.sarcasmSession.run({
      pooled_output: pooledOutput
    });
    
    // Step 4: Run intensity head
    const intensityResults = await this.intensitySession.run({
      pooled_output: pooledOutput
    });

    // Process results
    const emotionLogits = Array.from(emotionResults.emotion_logits.data);
    const sarcasmLogits = Array.from(sarcasmResults.sarcasm_logits.data);
    const intensityLogits = Array.from(intensityResults.intensity_logits.data);

    // Apply softmax to emotions
    const emotionProbs = this.softmax(emotionLogits);
    const sarcasmProb = this.sigmoid(sarcasmLogits[0]);
    const intensityProbs = this.softmax(intensityLogits);

    // Get top emotions
    const emotionIndices = emotionProbs
      .map((prob, idx) => ({ emotion: this.config.emotion_classes[idx], confidence: prob }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    // Determine intensity level from ordinal probabilities
    const intensityLabels = ['mild', 'primary', 'intense'];
    const maxIntensityIdx = intensityProbs.indexOf(Math.max(...intensityProbs));
    const intensityLabel = intensityLabels[maxIntensityIdx];

    const processingTime = performance.now() - startTime;

    return {
      emotions: emotionIndices,
      sarcasm_probability: sarcasmProb,
      intensity: intensityLabel,
      intensity_scores: { mild: intensityProbs[0], primary: intensityProbs[1], intense: intensityProbs[2] },
      primary_emotion: emotionIndices[0]?.emotion || 'unknown',
      privacy_mode: 'on_device',
      processing_time_ms: Math.round(processingTime * 100) / 100
    };
  }

  softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  dispose() {
    if (this.encoderSession) this.encoderSession.release();
    if (this.emotionSession) this.emotionSession.release();
    if (this.sarcasmSession) this.sarcasmSession.release();
    if (this.intensitySession) this.intensitySession.release();
    this.initialized = false;
  }
}

// Export for use in service worker or content script
export default PlutchikOnDeviceInference;
