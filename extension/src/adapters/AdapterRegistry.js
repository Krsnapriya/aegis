/**
 * Plutchik ERC — DOM Adapter Registry
 * Targets specific platforms for optimized extraction and injection.
 */

const RedditAdapter = {
  name: 'reddit',
  match: (url) => url.includes('reddit.com'),
  extract: () => {
    return [...document.querySelectorAll('[data-testid="comment"]')]
      .map(el => {
        const textEl = el.querySelector('p');
        return textEl ? { text: textEl.innerText.trim(), element: textEl } : null;
      })
      .filter(m => m && m.text.length >= 20);
  },
  modelHint: 'roberta',
  badgeTarget: (el) => el
};

const GmailAdapter = {
  name: 'gmail',
  match: (url) => url.includes('mail.google.com'),
  extract: () => {
    return [...document.querySelectorAll('.a3s.aiL, .gmail_quote')]
      .map(el => ({ text: el.innerText.trim(), element: el }))
      .filter(m => m.text.length >= 30);
  },
  modelHint: 'nemotron',
  badgeTarget: (el) => el
};

const LinkedInAdapter = {
  name: 'linkedin',
  match: (url) => url.includes('linkedin.com'),
  extract: () => {
    return [...document.querySelectorAll('.feed-shared-update-v2__description, .comment-item__main-content')]
      .map(el => ({ text: el.innerText.trim(), element: el }))
      .filter(m => m.text.length >= 30);
  },
  modelHint: 'nemotron',
  badgeTarget: (el) => el
};

const GenericAdapter = {
  name: 'generic',
  match: () => true,
  extract: () => {
    // Heuristic for any site: look for paragraphs in comment-like containers
    return [...document.querySelectorAll('p, .comment, [data-message]')]
      .map(el => ({ text: el.innerText.trim(), element: el }))
      .filter(m => m.text.length >= 50);
  },
  modelHint: 'auto',
  badgeTarget: (el) => el
};

export const AdapterRegistry = {
  adapters: [RedditAdapter, GmailAdapter, LinkedInAdapter, GenericAdapter],

  resolve(url) {
    return this.adapters.find(a => a.match(url)) || GenericAdapter;
  }
};
