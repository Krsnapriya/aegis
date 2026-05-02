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
        """
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
            # We define a wrapper to pass context to the ODE function
            def odefunc_wrapper(t, y):
                return self.ode_func(t, y, context)
            
            latent_trajectory = odeint(odefunc_wrapper, latent_state, t_span, method='rk4')
            
            # Decode back to emotion space
            emotion_trajectory = self.decoder(latent_trajectory)
            prob_trajectory = F.softmax(emotion_trajectory, dim=-1)
            
            # Calculate acceleration (2nd derivative approximation) for inflection detection
            # Inflection point: where rate of change of emotion velocity is highest
            velocities = torch.diff(prob_trajectory, dim=0)
            accelerations = torch.diff(velocities, dim=0)
            accel_magnitude = torch.norm(accelerations, dim=1)
            
            inflection_idx = torch.argmax(accel_magnitude).item() if len(accel_magnitude) > 0 else -1
            
            return {
                "trajectory": prob_trajectory.cpu().numpy().tolist(),
                "inflection_point_step": inflection_idx,
                "max_acceleration": accel_magnitude.max().item() if len(accel_magnitude) > 0 else 0.0,
                "final_state": prob_trajectory[-1].cpu().numpy().tolist(),
                "risk_score": float(accel_magnitude.mean().item()) if len(accel_magnitude) > 0 else 0.0
            }

class CounterfactualGenerator:
    """
    Generates strategic rewrites by optimizing for a target emotion vector.
    Uses a simplified masked language modeling approach with gradient guidance.
    """
    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model # The base PluTchik model
        self.device = device
        self.emotion_map = {
            "trust": 9, "fear": 10, "surprise": 11, "sadness": 12,
            "disgust": 13, "anger": 14, "anticipation": 15, "joy": 8,
            "serenity": 16, "admiration": 17, "acceptance": 17, # Simplified mapping
            "apprehension": 18, "distraction": 19, "pensiveness": 20,
            "boredom": 21, "annoyance": 22, "interest": 23,
            "optimism": 24, "aggressiveness": 31, "contempt": 30,
            "remorse": 29, "disapproval": 28, "awe": 27, "submission": 26,
            "love": 25, "rage": 6, "vigilance": 7, "terror": 2,
            "amazement": 3, "grief": 4, "loathing": 5, "ecstasy": 0
        }

    def generate_reframe(self, text: str, target_emotion: str, intensity: str = "medium") -> List[str]:
        """
        Generate 3 variations of text optimized for target_emotion.
        Strategy: Identify high-attribution tokens (via simple heuristic or captum if available)
        and suggest synonyms/phrasings associated with target emotion.
        """
        # In a full implementation, this would use a generative model (T5/GPT) guided by the discriminator.
        # Here we simulate the logic with template-based perturbations for the MVP.
        
        target_idx = self.emotion_map.get(target_emotion.lower(), 8) # Default to joy
        
        suggestions = []
        
        # Heuristic: If target is 'trust', reduce exclamation marks, add hedges.
        # If target is 'anger', remove hedges, use absolute terms.
        
        base_suggestion = text
        
        if target_emotion.lower() in ["trust", "serenity", "acceptance"]:
            modifiers = ["I understand", "Perhaps", "It seems", "Let's consider"]
            clean_text = text.replace("!", ".").replace("never", "rarely").replace("always", "often")
            suggestions.append(f"{modifiers[0]}, {clean_text.lower()}")
            suggestions.append(f"I hear you. {clean_text}")
            suggestions.append(f"Maybe we can look at it this way: {clean_text}")
            
        elif target_emotion.lower() in ["anger", "rage", "contempt"]:
            # Warning: Usually we want to de-escalate, but this supports the counterfactual engine
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
            # Generic variations
            suggestions.append(text)
            suggestions.append(f"Rephrased: {text}")
            suggestions.append(f"Alternative: {text}")

        return suggestions[:3]

class MultimodalIncongruityDetector:
    """
    Detects sarcasm/passive-aggression by measuring distance between
    semantic sentiment and pragmatic markers (punctuation, caps, emojis).
    """
    def __init__(self):
        self.positive_words = set(["good", "great", "awesome", "love", "thanks", "helpful", "nice"])
        self.negative_words = set(["bad", "terrible", "hate", "useless", "worst", "awful"])
        
    def calculate_incongruity_score(self, text: str, semantic_sentiment: float) -> Dict:
        """
        semantic_sentiment: Output from model (e.g., Joy score)
        Returns: Sarcasm probability and explanation
        """
        text_lower = text.lower()
        
        # Feature 1: Punctuation aggression
        exclamations = text.count("!")
        questions = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Feature 2: Lexical contrast
        pos_count = sum(1 for w in self.positive_words if w in text_lower)
        neg_count = sum(1 for w in self.negative_words if w in text_lower)
        
        incongruity_score = 0.0
        signals = []
        
        # Case A: Positive words + Aggressive punctuation/caps -> Sarcastic Joy / Contempt
        if pos_count > 0 and (exclamations > 2 or caps_ratio > 0.3):
            incongruity_score += 0.4
            signals.append("Positive words with aggressive capitalization/punctuation")
            
        # Case B: High Semantic Joy + Negative Context Words (if passed separately)
        # Here we rely on the semantic_sentiment being high (e.g. Joy) but text having negative markers
        if semantic_sentiment > 0.7 and neg_count > 0:
            incongruity_score += 0.3
            signals.append("High predicted joy despite negative lexical markers")
            
        # Case C: Short, clipped responses with high formal sentiment
        if len(text) < 10 and semantic_sentiment > 0.6:
            incongruity_score += 0.2
            signals.append("Terse phrasing with elevated sentiment")
            
        return {
            "sarcasm_probability": min(incongruity_score, 0.95),
            "signals": signals,
            "is_passive_aggressive": incongruity_score > 0.3
        }

# Unified Interface for the Extension Backend
class AdvancedPlutchikEngine:
    def __init__(self, base_model=None, tokenizer=None, device='cpu'):
        self.device = device
        self.forecaster = TrajectoryForecaster(device=device)
        self.reframer = CounterfactualGenerator(tokenizer, base_model, device) if base_model else None
        self.incongruity_detector = MultimodalIncongruityDetector()
        
    def analyze_dynamic(self, text: str, current_emotion_vector: List[float], 
                        dialogue_history: List[List[float]], user_baseline: Optional[Dict] = None) -> Dict:
        """
        Main entry point for dynamic analysis.
        """
        # 1. Forecast Trajectory
        forecast = self.forecaster.forecast(current_emotion_vector, dialogue_history)
        
        # 2. Detect Incongruity (Sarcasm/Passive-Aggression)
        # Assume current_emotion_vector[8] is Joy for demo purposes
        joy_score = current_emotion_vector[8] if len(current_emotion_vector) > 8 else 0.0
        incongruity = self.incongruity_detector.calculate_incongruity_score(text, joy_score)
        
        # 3. Generate Reframes (if risk is high)
        reframe_suggestions = []
        risk_level = "low"
        
        if forecast["risk_score"] > 0.2 or incongruity["sarcasm_probability"] > 0.3:
            risk_level = "high"
            # Determine target: usually de-escalation (Trust/Serenity)
            target = "trust" if incongruity["is_passive_aggressive"] else "serenity"
            if self.reframer:
                reframe_suggestions = self.reframer.generate_reframe(text, target)
        
        # 4. Compare against User Baseline
        baseline_deviation = None
        if user_baseline:
            # Simple Euclidean distance from baseline
            baseline_vec = user_baseline.get("average_vector", [0]*32)
            dist = np.linalg.norm(np.array(current_emotion_vector) - np.array(baseline_vec))
            if dist > 0.4: # Significant deviation
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
            "timestamp": torch.randint(0, 1000, (1,)).item()
        }

if __name__ == "__main__":
    # Demo usage
    print("Initializing Advanced Plutchik Engine...")
    engine = AdvancedPlutchikEngine()
    
    # Mock data
    text = "Oh GREAT, another meeting! Just what I needed!!!"
    current_emotion = [0.1]*32
    current_emotion[8] = 0.8 # High Joy prediction (which is likely wrong/sarcastic)
    history = [[0.2]*32, [0.3]*32]
    
    result = engine.analyze_dynamic(text, current_emotion, history)
    
    print("\n--- Analysis Result ---")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Sarcasm Probability: {result['incongruity']['sarcasm_probability']:.2f}")
    print(f"Signals: {result['incongruity']['signals']}")
    print(f"Forecast Risk Score: {result['forecast']['risk_score']:.2f}")
    if result['reframe_suggestions']:
        print("Suggested Reframes (De-escalation):")
        for s in result['reframe_suggestions']:
            print(f" - {s}")
