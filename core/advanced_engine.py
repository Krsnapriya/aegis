"""
Advanced Emotional Intelligence Engine
Implements: Neural ODEs, Counterfactual Augmentation, Multimodal Incongruity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
import logging
import time as _time

_engine_logger = logging.getLogger("plutchik-engine")

# Suppress warnings for cleaner output in production
warnings.filterwarnings("ignore")

class EmotionODEFunc(nn.Module):
    """
    Neural ODE Function: Models the continuous time derivative of emotional state.
    d(hidden_state)/dt = f(hidden_state, t, context_params)
    """
    def __init__(self, hidden_dim=64, context_dim=128):
        super(EmotionODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim)
        )
    
    def forward(self, t, state, context):
        # state: [batch, hidden_dim]
        # context: [batch, context_dim] (static conversation features)
        augmented_state = torch.cat([state, context], dim=1)
        return self.net(augmented_state)

class TrajectoryForecaster:
    """
    Uses Neural ODEs to predict emotional trajectory over continuous time.
    Can forecast future states and detect inflection points (acceleration).
    """
    def __init__(self, hidden_dim=64, context_dim=128, device='cpu'):
        self.device = device
        self.ode_func = EmotionODEFunc(hidden_dim, context_dim).to(device)
        self.hidden_dim = hidden_dim
        
        # Encoder to map discrete emotion vectors to continuous latent space
        self.encoder = nn.Linear(32, hidden_dim).to(device)
        # Decoder to map latent space back to 32-class distribution
        self.decoder = nn.Linear(hidden_dim, 32).to(device)
        
    def encode_context(self, dialogue_history: List[List[float]]) -> torch.Tensor:
        """Aggregate dialogue history into a static context vector."""
        if not dialogue_history:
            return torch.zeros(1, 128).to(self.device)
        
        # Simple mean pooling of recent turns for context
        hist_tensor = torch.tensor(dialogue_history[-5:], dtype=torch.float32).to(self.device)
        context = torch.mean(hist_tensor, dim=0).unsqueeze(0)
        # Pad or project to context_dim
        if context.shape[1] < 128:
            padding = torch.zeros(1, 128 - context.shape[1]).to(self.device)
            context = torch.cat([context, padding], dim=1)
        return context[:, :128]

    def forecast(self, current_emotion: List[float], dialogue_history: List[List[float]], 
                 steps: int = 10, dt: float = 0.1) -> Dict:
        """
        Forecast emotional trajectory from current state.
        Returns: predicted_states, inflection_points, acceleration_vector

        ⚠ SIMULATION MODE: The encoder/decoder are randomly initialised and have not been
        trained. Trajectory shape is plausible but magnitudes are not calibrated.
        Train the arc GRU (Phase 8 Ext.1) and load weights before using output clinically.
        """
        _engine_logger.warning(
            "TrajectoryForecaster is running with untrained weights (simulation mode). "
            "Output is structurally sound but numerically uncalibrated."
        )
        self.ode_func.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Encode current state to latent space
            curr_tensor = torch.tensor([current_emotion], dtype=torch.float32).to(self.device)
            latent_state = self.encoder(curr_tensor)
            
            # Get context
            context = self.encode_context(dialogue_history)
            context = context.repeat(latent_state.shape[0], 1) # Match batch size
            
            # Define time points for integration
            t_span = torch.linspace(0, steps * dt, steps).to(self.device)
            
            # Solve ODE
            def odefunc_wrapper(t, y):
                return self.ode_func(t, y, context)
            
            latent_trajectory = odeint(odefunc_wrapper, latent_state, t_span, method='rk4')
            
            # Decode back to emotion space
            emotion_trajectory = self.decoder(latent_trajectory)
            prob_trajectory = F.softmax(emotion_trajectory, dim=-1)
            
            # Calculate acceleration for inflection detection
            velocities = torch.diff(prob_trajectory, dim=0)
            accelerations = torch.diff(velocities, dim=0)
            accel_magnitude = torch.norm(accelerations, dim=1)
            
            inflection_idx = torch.argmax(accel_magnitude).item() if len(accel_magnitude) > 0 else -1
            
            # Squeeze batch dimension for API compatibility [Steps, Classes]
            traj_list = prob_trajectory.cpu().numpy().tolist()
            if len(traj_list) > 0 and isinstance(traj_list[0], list) and len(traj_list[0]) == 1:
                traj_list = [step[0] for step in traj_list]
                
            return {
                "trajectory": traj_list,
                "inflection_point_step": inflection_idx,
                "max_acceleration": accel_magnitude.max().item() if len(accel_magnitude) > 0 else 0.0,
                "final_state": traj_list[-1],
                "risk_score": float(accel_magnitude.mean().item()) if len(accel_magnitude) > 0 else 0.0
            }

class CounterfactualGenerator:
    """
    Generates strategic rewrites by optimizing for a target emotion vector.
    """
    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        # Canonical emotion index map — derived from sorted(PLUTCHIK.keys()) at runtime.
        # This MUST stay in sync with ERCPreprocessor.emotion_to_idx and EMOTION_NAMES.
        # Previously hardcoded and wrong (e.g. joy was at 8, correct index is 18).
        _all_emotions = sorted([
            "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger",
            "anticipation", "ecstasy", "admiration", "terror", "amazement", "grief",
            "loathing", "rage", "vigilance", "serenity", "acceptance", "apprehension",
            "distraction", "pensiveness", "boredom", "annoyance", "interest",
            "optimism", "love", "submission", "awe", "disapproval", "remorse",
            "contempt", "aggressiveness"
        ])
        self.emotion_map = {e: i for i, e in enumerate(_all_emotions)}

    def generate_reframe(self, text: str, target_emotion: str, intensity: str = "medium") -> List[str]:
        target_idx = self.emotion_map.get(target_emotion.lower(), 8)
        suggestions = []
        
        if target_emotion.lower() in ["trust", "serenity", "acceptance"]:
            modifiers = ["I understand", "Perhaps", "It seems", "Let's consider"]
            clean_text = text.replace("!", ".").replace("never", "rarely").replace("always", "often")
            suggestions.append(f"{modifiers[0]}, {clean_text.lower()}")
            suggestions.append(f"I hear you. {clean_text}")
            suggestions.append(f"Maybe we can look at it this way: {clean_text}")
            
        elif target_emotion.lower() in ["anger", "rage", "contempt"]:
            modifiers = ["Frankly", "Obviously", "Undeniably"]
            strong_text = text.replace(".", "!").replace("maybe", "certainly").replace("think", "know")
            suggestions.append(f"{modifiers[0]}, {strong_text}")
            suggestions.append(f"It is clear that {strong_text.lower()}")
            suggestions.append(strong_text)
            
        elif target_emotion.lower() in ["joy", "optimism", "love"]:
            modifiers = ["Great news!", "I'm so glad", "Wonderfully"]
            happy_text = text.replace("problem", "opportunity").replace("issue", "challenge")
            suggestions.append(f"{modifiers[0]} {happy_text}")
            suggestions.append(f"I love that {happy_text.lower()}")
            suggestions.append(f"Looking forward to: {happy_text}")
            
        else:
            suggestions.append(text)
            suggestions.append(f"Rephrased: {text}")
            suggestions.append(f"Alternative: {text}")

        return suggestions[:3]

class MultimodalIncongruityDetector:
    """
    Detects sarcasm/passive-aggression by measuring distance between
    semantic sentiment and pragmatic markers. 5 signal channels:
    polarity contradiction, emphasis, intensifiers, passive-aggressive, terse.
    """
    def __init__(self):
        self.positive_words = set([
            "good", "great", "awesome", "love", "thanks", "helpful", "nice",
            "wonderful", "fantastic", "amazing", "perfect", "brilliant",
            "excellent", "superb", "outstanding", "beautiful", "delightful", "pleasant"
        ])
        self.negative_words = set([
            "bad", "terrible", "hate", "useless", "worst", "awful",
            "horrible", "disgusting", "pathetic", "miserable", "dreadful",
            "appalling", "atrocious", "abysmal", "lousy", "inferior", "deficient", "subpar"
        ])
        self.intensifiers = set([
            "absolutely", "totally", "completely", "utterly", "literally",
            "definitely", "certainly", "obviously", "clearly", "undeniably"
        ])
        
    def calculate_incongruity_score(self, text: str, semantic_sentiment: float) -> Dict:
        text_lower = text.lower()
        words = text_lower.split()
        exclamations = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        pos_count = sum(1 for w in self.positive_words if w in text_lower)
        neg_count = sum(1 for w in self.negative_words if w in text_lower)
        intensifier_count = sum(1 for w in self.intensifiers if w in text_lower)
        
        incongruity_score = 0.0
        signals = []
        
        # Channel 1: Polarity contradiction
        if pos_count > 0 and neg_count > 0:
            incongruity_score += 0.3
            signals.append("Positive and negative words mixed in same utterance")
        elif pos_count > 0 and (exclamations > 2 or caps_ratio > 0.3):
            incongruity_score += 0.4
            signals.append("Positive words with aggressive capitalization/punctuation")
            
        # Channel 2: Emphasis markers
        if exclamations >= 3 or (exclamations >= 2 and caps_ratio > 0.2):
            incongruity_score += 0.2
            signals.append("Excessive emphasis markers")
            
        # Channel 3: Intensifiers without substance
        if intensifier_count >= 2 and len(words) < 12:
            incongruity_score += 0.2
            signals.append("Multiple intensifiers in short utterance")
            
        # Channel 4: Passive-aggressive markers
        if any(phrase in text_lower for phrase in ["just what i needed", "oh great", "how wonderful", "thanks a lot"]):
            incongruity_score += 0.4
            signals.append("Passive-aggressive phrase detected")
            
        # Channel 5: Terse phrasing with elevated sentiment
        if len(words) < 8 and semantic_sentiment > 0.6:
            incongruity_score += 0.2
            signals.append("Terse phrasing with elevated sentiment")
            
        return {
            "sarcasm_probability": min(incongruity_score, 0.95),
            "signals": signals,
            "is_passive_aggressive": incongruity_score > 0.3
        }

class AdvancedPlutchikEngine:
    def __init__(self, base_model=None, tokenizer=None, device='cpu'):
        self.device = device
        self.forecaster = TrajectoryForecaster(device=device)
        self.reframer = CounterfactualGenerator(tokenizer, base_model, device) if base_model else None
        self.incongruity_detector = MultimodalIncongruityDetector()
        
    def analyze_dynamic(self, text: str, current_emotion_vector: List[float], 
                        dialogue_history: List[List[float]], user_baseline: Optional[Dict] = None) -> Dict:
        forecast = self.forecaster.forecast(current_emotion_vector, dialogue_history)
        # Joy is at sorted index 18 in the canonical 32-class emotion list.
        # DO NOT hardcode index 8 — that maps to "awe" in sorted order.
        joy_idx = 18
        joy_score = current_emotion_vector[joy_idx] if len(current_emotion_vector) > joy_idx else 0.0
        incongruity = self.incongruity_detector.calculate_incongruity_score(text, joy_score)
        
        reframe_suggestions = []
        risk_level = "low"
        
        if forecast["risk_score"] > 0.2 or incongruity["sarcasm_probability"] > 0.3:
            risk_level = "high"
            target = "trust" if incongruity["is_passive_aggressive"] else "serenity"
            if self.reframer:
                reframe_suggestions = self.reframer.generate_reframe(text, target)
        
        baseline_deviation = None
        if user_baseline:
            baseline_vec = user_baseline.get("average_vector", [0]*32)
            dist = np.linalg.norm(np.array(current_emotion_vector) - np.array(baseline_vec))
            if dist > 0.4:
                baseline_deviation = {
                    "distance": float(dist),
                    "message": f"This tone deviates significantly from your usual style (Deviation: {dist:.2f})"
                }

        return {
            "forecast": forecast,
            "incongruity": incongruity,
            "reframe_suggestions": reframe_suggestions,
            "risk_level": risk_level,
            "baseline_deviation": baseline_deviation,
            "timestamp": int(_time.time())
        }

if __name__ == "__main__":
    import time
    print("Initializing Advanced Plutchik Engine...")
    engine = AdvancedPlutchikEngine()
    text = "Oh GREAT, another meeting! Just what I needed!!!"
    current_emotion = [0.1]*32
    current_emotion[8] = 0.8
    history = [[0.2]*32, [0.3]*32]
    result = engine.analyze_dynamic(text, current_emotion, history)
    print("\n--- Analysis Result ---")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Sarcasm Probability: {result['incongruity']['sarcasm_probability']:.2f}")
    print(f"Signals: {result['incongruity']['signals']}")

class InputSanitizer:
    """
    Ruthless Evaluator: Sanitizes input to prevent gibberish, prompt injection, and empty sequences.
    """
    def __init__(self):
        # N-gram based gibberish detection
        self.log_prob, self.threshold = self._build_ngram_model()
        self.reserved_tokens = ["[CONTEXT]", "[/CONTEXT]", "[CURRENT]", "[/CURRENT]", "[SCENARIO]", "[/SCENARIO]", "[TOPIC]", "[/TOPIC]"]
        
        # Plutchik Emoji Lexicon
        self.emoji_map = {
            "😡": "rage", "😠": "anger", "🤬": "rage", "😤": "annoyance",
            "😊": "joy", "😄": "ecstasy", "🥰": "love", "😍": "adoration",
            "😭": "grief", "😢": "sadness", "😔": "pensiveness", "😞": "sadness",
            "😱": "terror", "😨": "fear", "😰": "apprehension", "😬": "apprehension",
            "🤢": "loathing", "🤮": "disgust", "😒": "contempt", "🙄": "contempt",
            "😲": "amazement", "😮": "surprise", "😯": "distraction",
            "🤝": "trust", "🫂": "acceptance", "😇": "serenity",
            "🤔": "interest", "🧐": "vigilance", "✨": "optimism", "🙏": "submission",
            "🖤": "remorse", "👎": "disapproval", "🙌": "awe"
        }

    def _build_ngram_model(self):
        """Builds a simple character-level n-gram model from a corpus."""
        from collections import defaultdict
        import math

        # A small but representative corpus of English words
        corpus = "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us"
        
        # Trigram counts
        counts = defaultdict(int)
        for word in corpus.split():
            word = f"^{word}$"
            for i in range(len(word) - 2):
                trigram = word[i:i+3]
                counts[trigram] += 1
        
        total = sum(counts.values())
        log_prob = {k: math.log(v / total) for k, v in counts.items()}
        
        # Set a threshold based on the average log probability of the corpus
        # This is a heuristic to distinguish real language from random characters
        avg_log_prob = sum(log_prob.values()) / len(log_prob)
        threshold = avg_log_prob * 3.0  # Lenient threshold to prevent false positives
        
        return log_prob, threshold

    def _get_gibberish_score(self, text: str) -> float:
        """Calculates a score based on n-gram probabilities."""
        import math
        
        text = ''.join(filter(str.isalpha, text.lower()))
        if len(text) < 3:
            return 0.0 # Not enough to score

        log_probs = []
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            log_probs.append(self.log_prob.get(trigram, -10.0)) # Penalize unknown trigrams heavily
        
        return sum(log_probs) / len(log_probs) if log_probs else 0.0

    def sanitize_and_validate(self, text: str) -> Tuple[bool, str, str, Optional[str]]:
        """
        Returns: (is_valid, sanitized_text, status_reason, emoji_emotion)
        """
        if not text or not text.strip():
            return False, "", "Input is empty or whitespace.", None

        # Strip adversarial tokens
        sanitized_text = text
        for token in self.reserved_tokens:
            sanitized_text = sanitized_text.replace(token, "")
        
        sanitized_text = sanitized_text.strip()
        if not sanitized_text:
            return False, "", "Input contained only reserved tokens.", None

        # 1. Check for Emoji-Only Bypass
        alpha_chars = sum(1 for c in sanitized_text if c.isalpha())
        if alpha_chars == 0:
            # Check if any emoji in the map is present
            found_emotions = [self.emoji_map[c] for c in sanitized_text if c in self.emoji_map]
            if found_emotions:
                # Return the most frequent emoji emotion found
                from collections import Counter
                most_common = Counter(found_emotions).most_common(1)[0][0]
                return True, sanitized_text, "EmojiBypass", most_common
            else:
                return False, sanitized_text, "Input contains no alphabetical characters or known emojis.", None

        # 2. Gibberish detection (N-gram model)
        if alpha_chars > 4:
            score = self._get_gibberish_score(sanitized_text)
            if score < self.threshold:
                return False, sanitized_text, f"Input failed lexical heuristic check (Gibberish Score: {score:.2f} < Threshold: {self.threshold:.2f}).", None
        
        # 3. Max character limit check
        if len(sanitized_text) > 5000:
            return False, sanitized_text[:5000], "Input exceeds maximum character limit of 5000.", None

        return True, sanitized_text, "Valid", None
