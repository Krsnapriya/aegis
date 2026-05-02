
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
