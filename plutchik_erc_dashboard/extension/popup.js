/**
 * Plutchik Emotion Lens — Popup Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  const scenarioSelect = document.getElementById('scenario');
  const endpointInput = document.getElementById('endpoint');

  // Load saved settings
  chrome.storage.local.get(['scenario', 'endpoint'], (result) => {
    if (result.scenario) scenarioSelect.value = result.scenario;
    if (result.endpoint) endpointInput.value = result.endpoint;
  });

  // Check API health
  const checkHealth = async () => {
    try {
      const response = await fetch(`${endpointInput.value}/health`);
      const data = await response.json();
      if (data.status === 'healthy') {
        statusDot.classList.add('active');
        statusText.innerText = 'Connected to Engine';
      } else {
        statusDot.classList.remove('active');
        statusText.innerText = 'Engine Warming Up...';
      }
    } catch (e) {
      statusDot.classList.remove('active');
      statusText.innerText = 'Engine Offline';
    }
  };

  checkHealth();

  // Save settings on change
  scenarioSelect.addEventListener('change', () => {
    chrome.storage.local.set({ scenario: scenarioSelect.value });
  });

  endpointInput.addEventListener('change', () => {
    chrome.storage.local.set({ endpoint: endpointInput.value });
    checkHealth();
  });
});
