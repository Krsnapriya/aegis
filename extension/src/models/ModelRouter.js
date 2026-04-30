/**
 * Plutchik ERC — Model Routing Engine
 * Decides between RoBERTa and Nemotron based on 5 signals.
 */

export class ModelRouter {
  static select(site, texts, adapterHint = 'auto') {
    const avgWords = (items) => {
      if (items.length === 0) return 0;
      const total = items.reduce((acc, t) => acc + t.split(/\s+/).length, 0);
      return total / items.length;
    };

    const hasFormalMarkers = (items) => {
      const markers = ["dear", "regards", "sincerely", "per our", "attached", "best regards"];
      return items.some(t => markers.some(m => t.toLowerCase().includes(m)));
    };

    const majority = (arr) => {
      const counts = arr.reduce((acc, val) => {
        acc[val] = (acc[val] || 0) + 1;
        return acc;
      }, {});
      return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    };

    const signals = {
      // Signal 1: Site Identity
      siteScore: (site === 'gmail' || site === 'linkedin') ? 'nemotron' : 'roberta',

      // Signal 2: Text Length
      lengthScore: avgWords(texts) > 80 ? 'nemotron' : 'roberta',

      // Signal 3: Formal Register
      formalScore: hasFormalMarkers(texts) ? 'nemotron' : 'roberta',

      // Signal 4: Adapter Hint (Priority)
      hintScore: adapterHint !== 'auto' ? adapterHint : null,

      // Signal 5: Device Capability (Hard constraint)
      capScore: (typeof navigator !== 'undefined' && navigator.gpu) ? null : 'roberta'
    };

    // Debugging trace
    console.log("[Plutchik] Routing Signals:", signals);

    // Priority Order: Hint > Capability > (Majority of Site, Length, Formal)
    const decision = signals.hintScore
      ?? signals.capScore
      ?? majority([signals.siteScore, signals.lengthScore, signals.formalScore]);

    console.log(`[Plutchik] Decision: ${decision}`);
    return decision;
  }
}
