/**
 * Plutchik On-Device Inference Engine
 * Privacy-First: All processing happens in the browser using WebAssembly/WebGPU.
 * Zero data leaves the device.
 *
 * ONNX path (full accuracy):
 *   1. Run `python export_for_browser.py` to generate chrome_extension/model/*.onnx
 *   2. Download onnxruntime-web ESM build into chrome_extension/lib/ort.esm.min.js
 *      See chrome_extension/lib/INSTALL.md for instructions.
 *
 * Heuristic fallback (no ONNX required):
 *   When ort.esm.min.js is absent or ONNX models are missing the engine automatically
 *   uses a lightweight lexicon-based classifier. Results are still useful for
 *   sarcasm detection, trajectory forecasting, and reframe suggestions.
 */

// 32-class Plutchik emotion labels (must match models/multitask_emotion_model.py)
const EMOTION_CLASSES = [
  'ecstasy', 'admiration', 'terror', 'amazement', 'grief', 'loathing',
  'rage', 'vigilance', 'joy', 'trust', 'fear', 'surprise', 'sadness',
  'disgust', 'anger', 'anticipation', 'serenity', 'acceptance',
  'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance',
  'interest', 'optimism', 'love', 'submission', 'awe', 'disapproval',
  'remorse', 'contempt', 'aggressiveness'
];

// Minimal lexicon for heuristic inference – each entry is a list of trigger words.
// Index order matches EMOTION_CLASSES above.
const EMOTION_LEXICON = [
  /* ecstasy */      ['ecstatic', 'overjoyed', 'elated', 'euphoric', 'thrilled', 'exhilarated'],
  /* admiration */   ['admire', 'admiration', 'respect', 'impressive', 'outstanding', 'commend'],
  /* terror */       ['terrified', 'horrified', 'petrified', 'dread', 'horrifying', 'nightmare'],
  /* amazement */    ['amazing', 'astonishing', 'astounding', 'incredible', 'unbelievable', 'jaw'],
  /* grief */        ['grief', 'mourning', 'devastated', 'heartbroken', 'bereaved', 'bereavement'],
  /* loathing */     ['loathe', 'abhor', 'despise', 'revolting', 'repulsive', 'abomination'],
  /* rage */         ['rage', 'furious', 'enraged', 'livid', 'infuriated', 'seething', 'outraged'],
  /* vigilance */    ['vigilant', 'alert', 'watchful', 'cautious', 'attentive', 'on guard'],
  /* joy */          ['happy', 'joy', 'joyful', 'glad', 'cheerful', 'delighted', 'pleased', 'wonderful'],
  /* trust */        ['trust', 'reliable', 'honest', 'faithful', 'dependable', 'sincere', 'believe'],
  /* fear */         ['afraid', 'scared', 'fearful', 'frightened', 'nervous', 'worried', 'anxious'],
  /* surprise */     ['surprised', 'shocked', 'unexpected', 'startled', 'wow', 'oh my'],
  /* sadness */      ['sad', 'unhappy', 'sorrowful', 'depressed', 'melancholy', 'down', 'blue', 'miss'],
  /* disgust */      ['disgusted', 'gross', 'nasty', 'yuck', 'eww', 'horrible', 'awful'],
  /* anger */        ['angry', 'mad', 'upset', 'annoyed', 'frustrated', 'irritated', 'hate'],
  /* anticipation */ ['excited', 'anticipate', 'looking forward', 'eager', 'expect', 'hopeful', 'can\'t wait'],
  /* serenity */     ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'content', 'okay', 'fine'],
  /* acceptance */   ['accept', 'alright', 'understand', 'agree', 'tolerate', 'okay with'],
  /* apprehension */ ['apprehensive', 'uneasy', 'concern', 'troubled', 'unsure', 'hesitant'],
  /* distraction */  ['distracted', 'unfocused', 'scattered', 'confused', 'random', 'whatever'],
  /* pensiveness */  ['pensive', 'thoughtful', 'reflective', 'contemplating', 'wondering', 'musing'],
  /* boredom */      ['bored', 'boring', 'tedious', 'dull', 'monotonous', 'uninteresting', 'meh'],
  /* annoyance */    ['annoying', 'irritating', 'bothersome', 'nuisance', 'pesky', 'ugh', 'ugh'],
  /* interest */     ['interesting', 'curious', 'fascinated', 'intriguing', 'engaged', 'wonder'],
  /* optimism */     ['optimistic', 'hopeful', 'positive', 'bright side', 'will be fine', 'confident'],
  /* love */         ['love', 'adore', 'cherish', 'fond', 'affection', 'dear', 'precious', 'care'],
  /* submission */   ['submit', 'comply', 'defer', 'follow', 'obey', 'yield', 'surrender'],
  /* awe */          ['awe', 'awesome', 'majestic', 'magnificent', 'breathtaking', 'spectacular'],
  /* disapproval */  ['disapprove', 'wrong', 'unacceptable', 'inappropriate', 'should not', 'disagree'],
  /* remorse */      ['sorry', 'regret', 'remorse', 'guilty', 'apologize', 'apologise', 'mistake'],
  /* contempt */     ['contempt', 'scorn', 'disdain', 'condescend', 'look down', 'pathetic'],
  /* aggressiveness*/['aggressive', 'hostile', 'attack', 'fight', 'confront', 'challenge', 'demand']
];

// Try to load onnxruntime-web from a locally bundled copy.
// INSTALL: Download ort.esm.min.js to chrome_extension/lib/ (see lib/INSTALL.md).
let ort = null;
try {
  // import.meta.url resolves relative to ondevice-inference.js, so ./lib/ is correct.
  const ortModule = await import(new URL('./lib/ort.esm.min.js', import.meta.url).href);
  ort = ortModule;
  console.log('[Plutchik] onnxruntime-web loaded successfully');
} catch (_e) {
  console.warn('[Plutchik] onnxruntime-web not found – using heuristic fallback. ' +
    'See chrome_extension/lib/INSTALL.md to enable full ONNX inference.');
}

class PlutchikOnDeviceInference {
  constructor() {
    this.useONNX = false;
    this.initialized = false;
    this.config = null;
    // ONNX session objects
    this.encoderSession = null;
    this.emotionSession = null;
    this.sarcasmSession = null;
    this.intensitySession = null;
  }

  // ─── Public API ────────────────────────────────────────────────────────────

  async initialize() {
    if (this.initialized) return;
    console.log('[Plutchik] Initializing on-device inference engine...');

    if (ort) {
      try {
        await this._initONNX();
        this.useONNX = true;
        console.log('[Plutchik] ✓ ONNX inference ready (WebGPU/WASM)');
      } catch (e) {
        console.warn('[Plutchik] ONNX init failed:', e.message, '– falling back to heuristic');
        this.useONNX = false;
      }
    }

    if (!this.useONNX) {
      console.log('[Plutchik] ✓ Heuristic inference ready (no ONNX required)');
    }

    this.initialized = true;
  }

  /**
   * Full dynamic analysis: emotion classification, sarcasm detection,
   * trajectory forecasting (10 steps), and counterfactual reframes.
   * All processing happens in-browser. Zero network requests.
   *
   * @param {string} text - Draft text to analyse
   * @returns {Promise<object>} Analysis result
   */
  async analyze(text) {
    if (!this.initialized) await this.initialize();

    const startTime = performance.now();

    // Step 1: Classify emotions (ONNX or heuristic)
    const base = this.useONNX
      ? await this._runONNX(text)
      : this._heuristicClassify(text);

    // Step 2: Multimodal incongruity detection (sarcasm / passive-aggression)
    // Joy is index 8 – use its confidence as the semantic sentiment signal
    const joySentiment = base.emotions.find(e => e.emotion === 'joy')?.confidence ?? 0;
    const incongruity = this._detectIncongruity(text, joySentiment);

    // Step 3: Neural ODE-inspired trajectory forecast (10 steps)
    const emotionVector = EMOTION_CLASSES.map(
      name => base.emotions.find(e => e.emotion === name)?.confidence ?? 0
    );
    const trajectory = this._forecastTrajectory(emotionVector, 10);

    // Step 4: Risk level and counterfactual reframes
    const riskLevel = this._computeRiskLevel(trajectory.risk_score, incongruity.sarcasm_probability);
    const reframes = (riskLevel !== 'low')
      ? this._generateReframes(text, incongruity.is_passive_aggressive ? 'trust' : 'serenity')
      : [];

    const processingTime = performance.now() - startTime;

    return {
      emotions: base.emotions,
      primary_emotion: base.emotions[0]?.emotion ?? 'unknown',
      intensity: base.intensity,
      intensity_scores: base.intensity_scores,
      primary_emotion_ring: base.intensity, // alias kept for API compat
      sarcasm_probability: incongruity.sarcasm_probability,
      sarcasm_signals: incongruity.signals,
      is_passive_aggressive: incongruity.is_passive_aggressive,
      risk_level: riskLevel,
      trajectory_forecast: trajectory.steps,
      inflection_point: trajectory.inflection_point_step,
      reframe_suggestions: reframes,
      privacy_mode: 'on_device',
      inference_mode: this.useONNX ? 'onnx' : 'heuristic',
      processing_time_ms: Math.round(processingTime * 10) / 10
    };
  }

  dispose() {
    if (this.encoderSession) this.encoderSession.release();
    if (this.emotionSession) this.emotionSession.release();
    if (this.sarcasmSession) this.sarcasmSession.release();
    if (this.intensitySession) this.intensitySession.release();
    this.initialized = false;
    this.useONNX = false;
  }

  // ─── ONNX path ─────────────────────────────────────────────────────────────

  async _initONNX() {
    const getUrl = (path) => chrome.runtime.getURL(path);
    const opts = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' };

    // Load model config
    const cfgResp = await fetch(getUrl('model/config.json'));
    this.config = await cfgResp.json();

    this.encoderSession  = await ort.InferenceSession.create(getUrl('model/encoder.onnx'),      opts);
    this.emotionSession  = await ort.InferenceSession.create(getUrl('model/emotion_head.onnx'), opts);
    this.sarcasmSession  = await ort.InferenceSession.create(getUrl('model/sarcasm_head.onnx'), { executionProviders: ['webgpu', 'wasm'] });
    this.intensitySession= await ort.InferenceSession.create(getUrl('model/intensity_head.onnx'),{ executionProviders: ['webgpu', 'wasm'] });
  }

  async _runONNX(text) {
    const startTime = performance.now();
    const tokens = this._tokenize(text);

    const encoderInputs = {
      input_ids:      new ort.Tensor('int64', BigInt64Array.from(tokens),                          [1, tokens.length]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(tokens.map(id => id !== 0n ? 1n : 0n)), [1, tokens.length])
    };

    const encoderResults  = await this.encoderSession.run(encoderInputs);
    const pooled          = encoderResults.pooled_output;
    const emotionResults  = await this.emotionSession.run({ pooled_output: pooled });
    const sarcasmResults  = await this.sarcasmSession.run({ pooled_output: pooled });
    const intensityResults= await this.intensitySession.run({ pooled_output: pooled });

    const emotionProbs   = this._softmax(Array.from(emotionResults.emotion_logits.data));
    const intensityProbs = this._softmax(Array.from(intensityResults.intensity_logits.data));
    const emotionClasses = this.config?.emotion_classes ?? EMOTION_CLASSES;

    const emotions = emotionProbs
      .map((p, i) => ({ emotion: emotionClasses[i], confidence: p }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8);

    const intensityLabels = ['mild', 'primary', 'intense'];
    const intensityIdx    = intensityProbs.indexOf(Math.max(...intensityProbs));

    return {
      emotions,
      intensity: intensityLabels[intensityIdx],
      intensity_scores: { mild: intensityProbs[0], primary: intensityProbs[1], intense: intensityProbs[2] },
      processing_time_ms: performance.now() - startTime
    };
  }

  /** Simple hash-based tokeniser (placeholder for a real WordPiece tokeniser). */
  _tokenize(text) {
    const words = text.toLowerCase().split(/\s+/).slice(0, 127); // leave room for CLS
    const ids   = words.map(w => BigInt(this._hashToken(w) % 30000 + 1)); // avoid 0 (PAD)
    while (ids.length < 128) ids.push(0n); // pad
    return ids;
  }

  _hashToken(token) {
    let h = 0;
    for (let i = 0; i < token.length; i++) {
      h = Math.imul(31, h) + token.charCodeAt(i) | 0;
    }
    return Math.abs(h);
  }

  // ─── Heuristic path ────────────────────────────────────────────────────────

  _heuristicClassify(text) {
    const lower = text.toLowerCase();
    const words = lower.split(/\W+/).filter(Boolean);

    // Score each emotion class by lexicon hits
    const scores = EMOTION_LEXICON.map(lexItems =>
      lexItems.reduce((acc, kw) => acc + (lower.includes(kw) ? 1 : 0), 0)
    );

    // Pragmatic boost: all-caps words suggest intensity
    const capsRatio = text.split('').filter(c => c >= 'A' && c <= 'Z').length / Math.max(text.length, 1);
    const exclamations = (text.match(/!/g) || []).length;
    const questions    = (text.match(/\?/g) || []).length;

    if (capsRatio > 0.3 || exclamations > 1) {
      // Boost anger-adjacent emotions
      scores[6]  += 0.5; // rage
      scores[14] += 0.5; // anger
      scores[22] += 0.3; // annoyance
    }
    if (questions > 1) {
      scores[10] += 0.3; // fear
      scores[18] += 0.3; // apprehension
    }

    // If no lexicon hits at all, default to serenity (neutral-ish)
    const total = scores.reduce((a, b) => a + b, 0) || 1;
    const defaultBias = total === 1 ? 1 : 0; // 1 means we added the default
    if (total === 1) scores[16] += 1; // serenity

    const probs = this._softmax(scores.map(s => s * 5)); // scale for sharper distribution

    const emotions = probs
      .map((p, i) => ({ emotion: EMOTION_CLASSES[i], confidence: p }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8);

    // Intensity: map top-emotion confidence to mild/primary/intense
    const topConf = emotions[0].confidence;
    const intensity = topConf > 0.5 ? 'intense' : topConf > 0.25 ? 'primary' : 'mild';
    const iMap = { mild: 0, primary: 1, intense: 2 };
    const iProbs = [0, 0, 0];
    iProbs[iMap[intensity]] = 0.7;
    iProbs[(iMap[intensity] + 1) % 3] = 0.2;
    iProbs[(iMap[intensity] + 2) % 3] = 0.1;

    return {
      emotions,
      intensity,
      intensity_scores: { mild: iProbs[0], primary: iProbs[1], intense: iProbs[2] }
    };
  }

  // ─── Multimodal incongruity detector (sarcasm / passive-aggression) ────────

  _detectIncongruity(text, semanticJoy) {
    const lower = text.toLowerCase();
    const positiveWords = ['good', 'great', 'awesome', 'love', 'thanks', 'helpful', 'nice', 'wonderful'];
    const negativeWords = ['bad', 'terrible', 'hate', 'useless', 'worst', 'awful', 'horrible'];

    const posCount  = positiveWords.filter(w => lower.includes(w)).length;
    const negCount  = negativeWords.filter(w => lower.includes(w)).length;
    const excl      = (text.match(/!/g) || []).length;
    const capsRatio = text.split('').filter(c => c >= 'A' && c <= 'Z').length / Math.max(text.length, 1);

    let score   = 0;
    const signals = [];

    // Case A: positive words + aggressive caps / punctuation → sarcastic enthusiasm
    if (posCount > 0 && (excl > 2 || capsRatio > 0.3)) {
      score += 0.4;
      signals.push('Positive words with aggressive capitalisation/punctuation');
    }
    // Case B: high predicted joy but negative lexical markers
    if (semanticJoy > 0.7 && negCount > 0) {
      score += 0.3;
      signals.push('High predicted joy despite negative lexical markers');
    }
    // Case C: terse, clipped phrasing with elevated sentiment
    if (text.length < 10 && semanticJoy > 0.6) {
      score += 0.2;
      signals.push('Terse phrasing with elevated sentiment');
    }

    return {
      sarcasm_probability: Math.min(score, 0.95),
      signals,
      is_passive_aggressive: score > 0.3
    };
  }

  // ─── Neural-ODE-inspired trajectory forecaster ────────────────────────────

  /**
   * Simplified continuous-time forecast using mean-reversion dynamics.
   * This is a lightweight placeholder for the full Neural ODE implemented in
   * advanced_engine.py. It models emotions as decaying toward a neutral
   * uniform distribution with slight amplification of the dominant emotion.
   *
   * @param {number[]} emotionVector - 32-element probability distribution
   * @param {number} steps - forecast horizon (default 10)
   */
  _forecastTrajectory(emotionVector, steps = 10) {
    const dt     = 0.1;
    const decay  = 0.85; // mean-reversion rate per step
    const neutral = 1 / emotionVector.length;

    const trajectory = [];
    let state = [...emotionVector];

    for (let t = 0; t < steps; t++) {
      // d(state)/dt ≈ decay * (neutral - state)  [mean reversion]
      state = state.map((s, i) => {
        const ds = decay * (neutral - s);
        return Math.max(0, s + ds * dt);
      });
      // Re-normalise
      const sum = state.reduce((a, b) => a + b, 0);
      state = state.map(s => s / sum);
      trajectory.push([...state]);
    }

    // Detect inflection point: max Euclidean acceleration (2nd-order finite diff)
    let maxAccel = 0;
    let inflectionStep = -1;
    for (let i = 1; i < trajectory.length - 1; i++) {
      const accel = trajectory[i].reduce((acc, v, j) => {
        const vel1 = trajectory[i][j] - trajectory[i - 1][j];
        const vel2 = trajectory[i + 1][j] - trajectory[i][j];
        return acc + (vel2 - vel1) ** 2;
      }, 0);
      if (accel > maxAccel) { maxAccel = accel; inflectionStep = i; }
    }

    // Risk score: mean squared deviation from starting distribution
    const riskScore = trajectory.reduce((acc, step) => {
      const dev = step.reduce((s, v, i) => s + (v - emotionVector[i]) ** 2, 0);
      return acc + dev;
    }, 0) / trajectory.length;

    return {
      steps: trajectory,
      inflection_point_step: inflectionStep,
      max_acceleration: maxAccel,
      risk_score: riskScore
    };
  }

  // ─── Counterfactual reframe generator ─────────────────────────────────────

  /**
   * Generates 3 alternative phrasings that target Trust or Serenity while
   * preserving the core message intent.
   * TODO: Replace template heuristics with a fine-tuned T5/GPT model for
   * neural-quality reframing.
   *
   * @param {string} text - Original draft text
   * @param {string} targetEmotion - 'trust' | 'serenity'
   */
  _generateReframes(text, targetEmotion = 'serenity') {
    const suggestions = [];

    if (['trust', 'serenity', 'acceptance'].includes(targetEmotion)) {
      const clean = text.replace(/!/g, '.').replace(/\bnever\b/gi, 'rarely').replace(/\balways\b/gi, 'often');
      const lower = clean.charAt(0).toLowerCase() + clean.slice(1);
      suggestions.push(`I understand — ${lower}`);
      suggestions.push(`I hear you. ${clean}`);
      suggestions.push(`Maybe we can look at it this way: ${clean}`);
    } else if (['joy', 'optimism', 'love'].includes(targetEmotion)) {
      const uplifted = text.replace(/\bproblem\b/gi, 'opportunity').replace(/\bissue\b/gi, 'challenge');
      suggestions.push(`Great news! ${uplifted}`);
      suggestions.push(`I love that ${uplifted.toLowerCase()}`);
      suggestions.push(`Looking forward to: ${uplifted}`);
    } else {
      // Generic de-escalation
      const clean = text.replace(/!/g, '.').trim();
      suggestions.push(`Perhaps ${clean.toLowerCase()}`);
      suggestions.push(`It seems like ${clean.toLowerCase()}`);
      suggestions.push(`Let's consider: ${clean}`);
    }

    return suggestions.slice(0, 3);
  }

  // ─── Risk level ────────────────────────────────────────────────────────────

  _computeRiskLevel(trajectoryRiskScore, sarcasmProbability) {
    const combined = trajectoryRiskScore * 10 + sarcasmProbability;
    if (combined > 0.6) return 'high';
    if (combined > 0.25) return 'medium';
    return 'low';
  }

  // ─── Math utilities ────────────────────────────────────────────────────────

  _softmax(logits) {
    const max  = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  _sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
}

// Export for use in the background service worker (ES module context).
export default PlutchikOnDeviceInference;
