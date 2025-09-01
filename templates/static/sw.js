const CACHE_NAME = 'moodlab-v3';
const APP_SHELL = [
  '/',
  '/static/manifest.json',
  'https://cdn.jsdelivr.net/npm/chart.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js',
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL))
  );
});

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

self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);
  
  // Don't cache any API calls, POST requests, or auth-related requests
  if (req.method === 'POST' || 
      url.pathname.startsWith('/analyze') || 
      url.pathname.startsWith('/history') ||
      url.pathname.startsWith('/login') ||
      url.pathname.startsWith('/signup') ||
      url.pathname.startsWith('/logout') ||
      url.pathname.startsWith('/user') ||
      url.pathname.startsWith('/upgrade')) {
    event.respondWith(fetch(req));
    return;
  }

  // For navigation requests, try network first, then cache
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).catch(() => caches.match('/'))
    );
    return;
  }

  // For all other requests, use cache-first strategy
  event.respondWith(
    caches.match(req).then(cached => {
      // Return cached version if available
      if (cached) return cached;
      
      // Otherwise fetch from network
      return fetch(req).then(response => {
        // Don't cache non-successful responses
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }
        
        // Clone the response
        const responseToCache = response.clone();
        
        // Add to cache for future visits
        caches.open(CACHE_NAME).then(cache => {
          cache.put(req, responseToCache);
        });
        
        return response;
      });
    }).catch(() => {
      // Fallback for failed requests
      if (req.mode === 'navigate') return caches.match('/');
      return new Response('Network error happened', {
        status: 408,
        headers: { 'Content-Type': 'text/plain' }
      });
    })
  );
});