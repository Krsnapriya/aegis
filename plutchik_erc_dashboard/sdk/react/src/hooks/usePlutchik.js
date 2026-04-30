import { useState, useCallback } from 'react';

/**
 * Hook to interact with the Plutchik ERC API.
 * 
 * @param {Object} config 
 * @param {string} config.baseUrl - The URL of the Plutchik FastAPI server (default: http://localhost:8000)
 * @param {string} config.sessionId - Optional session identifier to maintain sliding context
 * @param {string} config.scenario - Contextual scenario for the model
 */
export function usePlutchik({ 
  baseUrl = 'http://localhost:8000', 
  sessionId = 'default-session',
  scenario = 'casual',
  topic = 'general'
} = {}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [arcTrajectory, setArcTrajectory] = useState(null);

  /**
   * Predict emotion for a single utterance
   */
  const predictEmotion = useCallback(async (text, speaker = 'USER') => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${baseUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          session_id: sessionId,
          speaker,
          scenario,
          topic
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      const data = await response.json();
      setPrediction(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [baseUrl, sessionId, scenario, topic]);

  /**
   * Submit a HITL correction back to the model
   */
  const submitCorrection = useCallback(async (text, predictedEmotion, correctedEmotion, confidence = 0, apiKey = '') => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${baseUrl}/correct`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({
          text,
          predicted_emotion: predictedEmotion,
          corrected_emotion: correctedEmotion,
          predicted_confidence: confidence,
          scenario
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [baseUrl, scenario]);

  /**
   * Clear current session memory
   */
  const clearSession = useCallback(async () => {
    try {
      await fetch(`${baseUrl}/session/${sessionId}`, {
        method: 'DELETE'
      });
      setPrediction(null);
    } catch (err) {
      console.error('Failed to clear session', err);
    }
  }, [baseUrl, sessionId]);

  /**
   * Analyze conversation arc
   */
  const predictArc = useCallback(async (utterances) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${baseUrl}/predict/arc`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          utterances,
          scenario,
          topic
        })
      });

      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);

      const data = await response.json();
      setArcTrajectory(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [baseUrl, scenario, topic]);

  return {
    predictEmotion,
    predictArc,
    submitCorrection,
    clearSession,
    prediction,
    arcTrajectory,
    loading,
    error
  };
}
