/**
 * Plutchik ERC — Content Script
 * Coordinates scanning, message passing, and badge injection.
 */

// We can't use 'import' directly in content scripts without bundling or using a loader.
// For this scaffold, we'll assume a build step or use dynamic imports if configured.
// Since we are writing raw files, I'll use a functional approach that assumes
// the registry and injector are available or bundled.

(async () => {
  // Dynamic import for ES modules in manifest v3 content scripts
  const { AdapterRegistry } = await import(chrome.runtime.getURL('src/adapters/AdapterRegistry.js'));
  const { injectBadge } = await import(chrome.runtime.getURL('src/ui/ShadowDom.js'));

  const adapter = AdapterRegistry.resolve(window.location.href);
  console.log(`[Plutchik] Adapter resolved: ${adapter.name}`);

  async function scanPage() {
    const items = adapter.extract();
    if (items.length === 0) return;

    console.log(`[Plutchik] Extracted ${items.length} items. Sending to background...`);

    chrome.runtime.sendMessage({
      type: 'SCAN_BATCH',
      items: items.map(i => i.text),
      site: adapter.name,
      modelHint: adapter.modelHint
    }, (response) => {
      if (response && response.predictions) {
        response.predictions.forEach((pred, idx) => {
          if (items[idx]) {
            const target = adapter.badgeTarget(items[idx].element);
            injectBadge(target, pred);
          }
        });
      }
    });
  }

  // Initial scan
  scanPage();

  // Observer for dynamic content (infinite scroll)
  let scanTimeout;
  const observer = new MutationObserver(() => {
    clearTimeout(scanTimeout);
    scanTimeout = setTimeout(scanPage, 1500); // Debounce
  });

  observer.observe(document.body, { childList: true, subtree: true });
})();
