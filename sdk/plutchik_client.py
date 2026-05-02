"""
Plutchik ERC Python SDK
Easy integration for any Python application
"""
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    sarcasm: bool
    sarcasm_score: float
    intensity: str
    intensity_scores: Dict[str, float]

class PlutchikClient:
    """
    Python client for Plutchik Emotion Recognition API
    
    Usage:
        client = PlutchikClient(api_url="http://localhost:8000", api_key="your-key")
        result = client.predict("I'm so excited about this!")
        print(f"Emotion: {result.emotion}, Confidence: {result.confidence:.1%}")
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: str = "demo-key-123"):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    def predict(self, text: str, context: Optional[List[str]] = None) -> EmotionResult:
        """Analyze a single utterance"""
        payload = {"text": text}
        if context:
            payload["context"] = context
        
        response = requests.post(
            f"{self.api_url}/predict",
            json=payload,
            headers=self._headers
        )
        response.raise_for_status()
        data = response.json()
        
        return EmotionResult(
            emotion=data['emotion'],
            confidence=data['confidence'],
            all_emotions=data['all_emotions'],
            sarcasm=data['sarcasm'],
            sarcasm_score=data['sarcasm_score'],
            intensity=data['intensity'],
            intensity_scores=data['intensity_scores']
        )
    
    def predict_batch(self, texts: List[str]) -> List[EmotionResult]:
        """Analyze multiple texts in batch"""
        response = requests.post(
            f"{self.api_url}/predict/batch",
            json=texts,
            headers=self._headers
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for pred in data['predictions']:
            results.append(EmotionResult(**pred))
        return results
    
    def predict_arc(self, conversation: List[str]) -> Dict[str, Any]:
        """Analyze emotion trajectory across a conversation"""
        response = requests.post(
            f"{self.api_url}/predict/arc",
            json=conversation,
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()
    
    def compare(self, conv_a: List[str], conv_b: List[str]) -> Dict[str, Any]:
        """Compare two conversations side-by-side"""
        response = requests.post(
            f"{self.api_url}/compare",
            json={"conv_a": conv_a, "conv_b": conv_b},
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()
    
    def explain(self, text: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get prediction with Captum token attributions"""
        params = {"text": text}
        if context:
            params["context"] = context
        
        response = requests.post(
            f"{self.api_url}/explain",
            json=params,
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check API health"""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()


# Convenience function for quick usage
def analyze(text: str, api_url: str = "http://localhost:8000") -> EmotionResult:
    """Quick one-liner emotion analysis"""
    client = PlutchikClient(api_url=api_url)
    return client.predict(text)


if __name__ == "__main__":
    # Demo
    client = PlutchikClient()
    
    # Single prediction
    result = client.predict("I'm absolutely thrilled about this opportunity!")
    print(f"Emotion: {result.emotion} ({result.confidence:.1%})")
    print(f"Sarcasm: {result.sarcasm} ({result.sarcasm_score:.1%})")
    print(f"Intensity: {result.intensity}")
    
    # Conversation arc
    conversation = [
        "I'm excited to start this project.",
        "Me too! The possibilities are endless.",
        "But I'm worried about the deadline.",
        "Don't worry, we've got this.",
        "Actually, I'm starting to panic now."
    ]
    
    arc = client.predict_arc(conversation)
    print(f"\nConversation Arc: {' → '.join(arc['emotional_arc'])}")
    print(f"Inflection points: {len(arc['inflection_points'])}")
