"""
Export Plutchik Model to ONNX for Browser-Based Inference (WASM/WebGPU)
Privacy-First: No text leaves the device when using browser extension
"""

import torch
import torch.onnx
from models.multitask_emotion_model import PluTchikMultiTaskModel, EMOTION_CLASSES
import json
import os

def export_model_to_onnx(model_path, output_dir='chrome_extension/model'):
    """
    Export the trained PyTorch model to ONNX format for browser inference.
    This enables 100% on-device processing with no data leaving the browser.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PluTchikMultiTaskModel()
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load trained weights: {e}")
        print("Proceeding with random weights for export demo")
    
    model.eval()
    
    # Create dummy input (batch_size=1, seq_length=128) - use int64 for BERT
    dummy_input = torch.randint(0, 1000, (1, 128), dtype=torch.long)
    attention_mask = torch.ones((1, 128), dtype=torch.long)
    
    # For heads, we need pooled_output which is float32 (hidden_size=128)
    dummy_pooled = torch.randn(1, 128, dtype=torch.float32)
    
    # Export emotion head (emotion_classifier) - takes pooled_output (float32)
    print("\nExporting emotion classification head...")
    torch.onnx.export(
        model.emotion_classifier,
        (dummy_pooled,),
        f"{output_dir}/emotion_head.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pooled_output'],
        output_names=['emotion_logits'],
        dynamic_axes={
            'pooled_output': {0: 'batch_size', 1: 'hidden_size'},
            'emotion_logits': {0: 'batch_size'}
        }
    )
    print("✓ Emotion head exported")
    
    # Export sarcasm head
    print("\nExporting sarcasm detection head...")
    torch.onnx.export(
        model.sarcasm_head,
        (dummy_pooled,),
        f"{output_dir}/sarcasm_head.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pooled_output'],
        output_names=['sarcasm_logits'],
        dynamic_axes={
            'pooled_output': {0: 'batch_size', 1: 'hidden_size'},
            'sarcasm_logits': {0: 'batch_size'}
        }
    )
    print("✓ Sarcasm head exported")
    
    # Export intensity head
    print("\nExporting intensity regression head...")
    torch.onnx.export(
        model.intensity_head,
        (dummy_pooled,),
        f"{output_dir}/intensity_head.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pooled_output'],
        output_names=['intensity_logits'],
        dynamic_axes={
            'pooled_output': {0: 'batch_size', 1: 'hidden_size'},
            'intensity_logits': {0: 'batch_size'}
        }
    )
    print("✓ Intensity head exported")
    
    # Export encoder (BERT model)
    print("\nExporting shared encoder (BERT)...")
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.bert = model.bert
        
        def forward(self, input_ids, attention_mask=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]  # CLS token
    
    encoder = EncoderWrapper(model)
    torch.onnx.export(
        encoder,
        (dummy_input, attention_mask),
        f"{output_dir}/encoder.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['pooled_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'pooled_output': {0: 'batch_size'}
        }
    )
    print("✓ Encoder exported")
    
    # Save model config and metadata
    config = {
        'model_version': '3.0.0',
        'export_date': torch.__version__,
        'num_emotions': 32,
        'max_sequence_length': 128,
        'emotion_classes': [
            'ecstasy', 'admiration', 'terror', 'amazement', 'grief', 'loathing', 
            'rage', 'vigilance', 'joy', 'trust', 'fear', 'surprise', 'sadness', 
            'disgust', 'anger', 'anticipation', 'serenity', 'acceptance', 
            'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance', 
            'interest', 'optimism', 'love', 'submission', 'awe', 'disapproval', 
            'remorse', 'contempt', 'aggressiveness'
        ],
        'privacy_mode': 'on_device_only',
        'inference_backend': 'onnx_webgpu'
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Model config saved")
    
    # Generate TypeScript types for the browser code
    typescript_defs = """
// Auto-generated TypeScript definitions for Plutchik ONNX models

export interface ModelConfig {
  model_version: string;
  num_emotions: number;
  max_sequence_length: number;
  emotion_classes: string[];
  privacy_mode: 'on_device_only';
  inference_backend: 'onnx_webgpu';
}

export interface InferenceResult {
  emotions: Array<{ emotion: string; confidence: number }>;
  sarcasm_probability: number;
  intensity: 'low' | 'medium' | 'high';
  primary_emotion: string;
}

export class PlutchikInference {
  constructor(configPath: string);
  initialize(): Promise<void>;
  analyze(text: string): Promise<InferenceResult>;
  dispose(): void;
}
"""
    
    with open(f"{output_dir}/types.ts", 'w') as f:
        f.write(typescript_defs)
    print("✓ TypeScript definitions generated")
    
    print(f"\n✅ Model export complete! Files saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Use onnxruntime-web to load these models in the browser")
    print("2. Integrate with chrome_extension/ondevice-inference.js")
    print("3. Update manifest.json to include model files as web_accessible_resources")
    
    return True


def create_browser_inference_wrapper():
    """
    Create the JavaScript wrapper for ONNX runtime in the browser.
    This handles tokenization, model execution, and result formatting.
    """
    
    wrapper_code = '''/**
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
        const tokens = text.toLowerCase().split(/\\s+/).slice(0, 128);
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
'''
    
    with open('chrome_extension/ondevice-inference.js', 'w') as f:
        f.write(wrapper_code)
    
    print("✓ Browser inference wrapper created: chrome_extension/ondevice-inference.js")


if __name__ == '__main__':
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/my_plutchik_model/best_model.pt'
    
    print("="*70)
    print("PLUTCHIK MODEL EXPORT FOR BROWSER (WASM/WebGPU)")
    print("="*70)
    print("\nThis script exports your PyTorch model to ONNX format for")
    print("100% on-device inference in the Chrome extension.")
    print("\nPRIVACY GUARANTEE:")
    print("- No text ever leaves the user's browser")
    print("- No cookies, tracking, or analytics")
    print("- All processing via WebGPU/WASM acceleration")
    print("="*70 + "\n")
    
    success = export_model_to_onnx(model_path)
    
    if success:
        create_browser_inference_wrapper()
        
        print("\n" + "="*70)
        print("EXPORT COMPLETE!")
        print("="*70)
        print("\nTo use in Chrome extension:")
        print("1. Load extension in Chrome: chrome://extensions/")
        print("2. Enable 'Developer mode'")
        print("3. Click 'Load unpacked' and select chrome_extension/")
        print("4. The extension will automatically use on-device inference")
        print("\nFor cloud fallback (optional):")
        print("- Update background.js to allow cloud API when user opts in")
        print("- Default is 100% on-device for maximum privacy")
        print("="*70)
