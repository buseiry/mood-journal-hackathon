const CACHE_NAME = 'moodlab-v3';
const APP_SHELL = [
  '/',
  '/static/manifest.json',
  'https://cdn.jsdelivr.net/npm/chart.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js',
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap'
];

// Install event: cache app shell
self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL))
  );
});

// Activate event: clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      caches.keys().then(keys =>
        Promise.all(
          keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
        )
      )
    ])
  );
});

// Fetch event: handle requests
self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);

  // Do not cache API or auth requests
  const noCachePaths = ['/analyze', '/history', '/login', '/signup', '/logout', '/user'];
  if (req.method === 'POST' || noCachePaths.some(path => url.pathname.startsWith(path))) {
    event.respondWith(fetch(req));
    return;
  }

  // Navigation requests: network-first, fallback to cache
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).catch(() => caches.match('/'))
    );
    return;
  }

  // Other requests: cache-first strategy
  event.respondWith(
    caches.match(req).then(cached => {
      if (cached) return cached;

      return fetch(req).then(response => {
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        const responseToCache = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(req, responseToCache));

        return response;
      });
    }).catch(() => {
      if (req.mode === 'navigate') return caches.match('/');
      return new Response('Network error happened', {
        status: 408,
        headers: { 'Content-Type': 'text/plain' }
      });
    })
  );
});
