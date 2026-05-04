import React, { useState } from 'react';
import { usePlutchik } from '../hooks/usePlutchik';

// Ring colors matching the Plutchik taxonomy
const RING_COLORS = {
  intense: '#ff7b72',
  primary: '#58a6ff',
  mild: '#a371f7',
  dyadic: '#3fb950',
};

/**
 * Drop-in React widget for Plutchik Emotion Analysis.
 */
export function PlutchikWidget({ 
  baseUrl = 'http://localhost:8000',
  sessionId = 'react-widget-session',
  scenario = 'casual',
  theme = 'dark'
}) {
  const [text, setText] = useState('');
  const { predictEmotion, prediction, loading, error, submitCorrection, clearSession } = usePlutchik({ 
    baseUrl, sessionId, scenario 
  });
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctionEmotion, setCorrectionEmotion] = useState('');

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setShowCorrection(false);
    await predictEmotion(text);
  };

  const handleCorrectionSubmit = async () => {
    if (!correctionEmotion || !prediction) return;
    await submitCorrection(
      prediction.text || text, 
      prediction.emotion, 
      correctionEmotion, 
      prediction.confidence
    );
    setShowCorrection(false);
    setCorrectionEmotion('');
    alert('Correction submitted to HITL queue!');
  };

  const isDark = theme === 'dark';
  
  const styles = {
    container: {
      fontFamily: 'Inter, system-ui, sans-serif',
      background: isDark ? '#161b22' : '#ffffff',
      color: isDark ? '#e6edf3' : '#24292e',
      border: `1px solid ${isDark ? '#30363d' : '#e1e4e8'}`,
      borderRadius: '8px',
      padding: '16px',
      width: '100%',
      maxWidth: '400px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
    },
    input: {
      width: '100%',
      padding: '8px 12px',
      borderRadius: '6px',
      border: `1px solid ${isDark ? '#30363d' : '#e1e4e8'}`,
      background: isDark ? '#0d1117' : '#f6f8fa',
      color: isDark ? '#c9d1d9' : '#24292e',
      marginBottom: '8px'
    },
    button: {
      background: '#238636',
      color: '#fff',
      border: 'none',
      padding: '6px 16px',
      borderRadius: '6px',
      cursor: 'pointer',
      fontWeight: '600'
    },
    emotionTag: {
      display: 'inline-block',
      padding: '4px 10px',
      borderRadius: '12px',
      fontSize: '14px',
      fontWeight: 'bold',
      textTransform: 'capitalize'
    }
  };

  return (
    <div style={styles.container}>
      <div style={{ marginBottom: '16px', fontWeight: 'bold' }}>🎭 Plutchik Emotion Analysis</div>
      
      <form onSubmit={handlePredict}>
        <input 
          type="text" 
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type a message..."
          style={styles.input}
        />
        <div style={{ display: 'flex', gap: '8px' }}>
          <button type="submit" disabled={loading} style={styles.button}>
            {loading ? 'Analyzing...' : 'Analyze Emotion'}
          </button>
          <button type="button" onClick={clearSession} style={{...styles.button, background: '#d73a49'}}>
            Clear Context
          </button>
        </div>
      </form>

      {error && <div style={{ color: '#ff7b72', marginTop: '12px', fontSize: '14px' }}>Error: {error}</div>}

      {prediction && (
        <div style={{ marginTop: '20px', padding: '12px', background: isDark ? '#0d1117' : '#f6f8fa', borderRadius: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ 
              ...styles.emotionTag, 
              color: RING_COLORS[prediction.ring] || '#8b949e',
              background: `${RING_COLORS[prediction.ring]}20` || 'transparent'
            }}>
              {prediction.emotion} ({(prediction.confidence * 100).toFixed(0)}%)
            </span>
            {prediction.sarcasm_prob > 0.5 && (
              <span style={{ fontSize: '12px', color: '#f0883e', border: '1px solid #f0883e', padding: '2px 6px', borderRadius: '4px' }}>
                Sarcastic
              </span>
            )}
          </div>
          
          <div style={{ fontSize: '12px', color: isDark ? '#8b949e' : '#586069', marginTop: '8px' }}>
            <strong>Intensity:</strong> {prediction.intensity.toFixed(2)} (Ring: {prediction.ring})<br/>
            <strong>Context:</strong> {prediction.context_used}
          </div>

          {!showCorrection ? (
            <button 
              onClick={() => setShowCorrection(true)}
              style={{ background: 'transparent', border: 'none', color: '#58a6ff', cursor: 'pointer', fontSize: '12px', marginTop: '12px', padding: 0 }}
            >
              Feedback: Was this wrong?
            </button>
          ) : (
            <div style={{ marginTop: '12px', borderTop: `1px solid ${isDark ? '#30363d' : '#e1e4e8'}`, paddingTop: '12px' }}>
              <div style={{ fontSize: '12px', marginBottom: '8px' }}>Suggest correct emotion:</div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input 
                  type="text" 
                  value={correctionEmotion}
                  onChange={(e) => setCorrectionEmotion(e.target.value.toLowerCase())}
                  placeholder="e.g. joy, anger..."
                  style={{...styles.input, marginBottom: 0}}
                />
                <button type="button" onClick={handleCorrectionSubmit} style={{...styles.button, background: '#1f6feb'}}>
                  Submit
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
