from pathlib import Path
import sys

_dash = Path(__file__).resolve().parent.parent
if str(_dash) not in sys.path:
    sys.path.insert(0, str(_dash))

import requests
import json
import os
from dotenv import load_dotenv
from utils.constants import EMOTION_NAMES

# Load environment variables. Try multiple paths for robustness.
_repo_root = Path(__file__).resolve().parent.parent
env_path = _repo_root / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

class NemotronClient:
    def __init__(self, model="nvidia/nemotron-3-super-120b-a12b:free"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def predict_emotion(self, text, scenario="general", topic="general", context="[NO_CONTEXT]"):
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            return self._mock_inference(text)

        prompt = f"""
        You are an expert in Plutchik's Wheel of Emotions and Emotion Recognition in Conversation (ERC).
        Analyze the following utterance and provide a structured JSON response.

        ### PLUTCHIK CATEGORIES (32):
        joy, trust, fear, surprise, sadness, disgust, anger, anticipation, 
        ecstasy, admiration, terror, amazement, grief, loathing, rage, vigilance, 
        serenity, acceptance, apprehension, distraction, pensiveness, boredom, annoyance, interest, 
        optimism, love, submission, awe, disapproval, remorse, contempt, aggressiveness.

        ### INPUT:
        - Scenario: {scenario}
        - Topic: {topic}
        - Context: {context}
        - Utterance: "{text}"

        ### TASK:
        1. Identify the most dominant emotion from the 32 categories above.
        2. Detect if there is sarcasm (subtext diverging from literal text).
        3. Rate the intensity of the emotion (0.0 to 1.0).
        4. Provide a brief one-sentence reasoning.

        ### OUTPUT FORMAT (JSON ONLY):
        {{
            "emotion": "string",
            "sarcasm_detected": boolean,
            "sarcasm_confidence": float (0.0 to 1.0),
            "intensity": float (0.0 to 1.0),
            "reasoning": "string"
        }}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Krsnapriya/plutchik",
            "X-Title": "Plutchik ERC Dashboard"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        print(f"Sending request to OpenRouter with model: {self.model}")
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            # Basic JSON extraction in case the model adds extra text
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                return {"error": "Failed to parse JSON: " + content}
            
            parsed = json.loads(content[start:end])
            if "emotion" in parsed and isinstance(parsed["emotion"], str):
                parsed["emotion"] = parsed["emotion"].lower()
                if parsed["emotion"] not in EMOTION_NAMES:
                    return {"error": f"Invalid emotion predicted by LLM: {parsed['emotion']}"}
            return parsed
            
        except requests.exceptions.Timeout:
            print("LLM API Timeout. Falling back to mock mode.")
            return self._mock_inference(text, warning="LLM API Timeout: Using offline fallback heuristics.")
        except requests.exceptions.HTTPError as e:
            err_text = response.text if 'response' in locals() else str(e)
            print(f"LLM API Error: {err_text}. Falling back to mock mode.")
            return self._mock_inference(text, warning=f"LLM Auth/API Error. Using offline fallback heuristics.")
        except Exception as e:
            print(f"LLM Connection Error: {str(e)}. Falling back to mock mode.")
            return self._mock_inference(text, warning="LLM Connection Error. Using offline fallback heuristics.")

    def _mock_inference(self, text, warning="Local Offline Mode (Mock LLM) Active."):
        # Simple heuristic fallback
        text_lower = text.lower()
        emotion = "neutral"
        if any(w in text_lower for w in ["happy", "great", "excellent", "good"]):
            emotion = "joy"
        elif any(w in text_lower for w in ["sad", "terrible", "bad", "depressed"]):
            emotion = "sadness"
        elif any(w in text_lower for w in ["angry", "mad", "furious"]):
            emotion = "anger"
        elif any(w in text_lower for w in ["scared", "fear", "terrified"]):
            emotion = "fear"
        else:
            emotion = "interest"
            
        sarcasm = "?" in text and "!" in text
        
        return {
            "emotion": emotion,
            "sarcasm_detected": sarcasm,
            "sarcasm_confidence": 0.8 if sarcasm else 0.1,
            "intensity": 0.6,
            "reasoning": f"[MOCK] Detected keyword indicators mapping to {emotion}.",
            "warning": warning
        }

if __name__ == "__main__":
    client = NemotronClient()
    test_text = "I am so happy that my flight got cancelled and I'm stuck at the airport for 12 hours."
    print(f"Testing Nemotron-3 with: {test_text}")
    print(json.dumps(client.predict_emotion(test_text, scenario="travel", topic="flight_delay"), indent=4))
