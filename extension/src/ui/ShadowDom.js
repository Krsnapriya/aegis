/**
 * Plutchik ERC — Shadow DOM Injection Engine
 * Prevents style bleed and ensures consistent badge appearance.
 */

export function injectBadge(element, prediction) {
  if (element.querySelector('.plutchik-host')) return; // Already injected

  const host = document.createElement('span');
  host.className = 'plutchik-host';
  host.style.cssText = 'display:inline-block; margin-left:8px; vertical-align:middle; cursor:default;';

  // Closed shadow root for absolute isolation
  const shadow = host.attachShadow({ mode: 'closed' });

  // Stylesheet for the shadow root
  const style = document.createElement('style');
  style.textContent = `
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 12px;
      font-family: 'Inter', system-ui, sans-serif;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      white-space: nowrap;
      transition: all 0.2s ease;
    }
    .badge:hover {
      transform: scale(1.05);
      filter: brightness(1.1);
    }
    .mild { background: #a371f722; color: #a371f7; border: 1px solid #a371f744; }
    .primary { background: #58a6ff22; color: #58a6ff; border: 1px solid #58a6ff44; }
    .intense { background: #ff7b7222; color: #ff7b72; border: 1px solid #ff7b7244; }
    .dyadic { background: #3fb95022; color: #3fb950; border: 1px solid #3fb95044; }
    .low-conf { 
      background: transparent; 
      color: #8b949e; 
      border: 1px dashed #30363d; 
      font-style: italic;
    }
    .tooltip {
      visibility: hidden;
      position: absolute;
      background: #161b22;
      color: #e6edf3;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid #30363d;
      z-index: 10000;
      font-size: 10px;
    }
  `;

  const badge = document.createElement('span');
  const ringClass = prediction.ring ? prediction.ring.toLowerCase() : 'primary';
  
  if (prediction.confidence < 0.55) {
    badge.textContent = 'low confidence';
    badge.className = 'badge low-conf';
  } else {
    badge.textContent = `${prediction.emotion} · ${Math.round(prediction.confidence * 100)}%`;
    badge.className = `badge ${ringClass}`;
  }

  shadow.appendChild(style);
  shadow.appendChild(badge);
  element.appendChild(host);
}
