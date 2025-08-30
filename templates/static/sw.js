const CACHE_NAME = 'moodlab-v1';
const APP_SHELL = [
  '/',
  '/static/manifest.json',
  '/static/sw.js',
  'https://cdn.jsdelivr.net/npm/chart.js'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL)));
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))));
});

self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);

  if (req.method === 'POST') return;

  if (url.pathname === '/history') {
    event.respondWith(fetch(req).then(res => { caches.open(CACHE_NAME).then(cache => cache.put(req, res.clone())); return res; }).catch(() => caches.match(req).then(cached => cached || caches.match('/'))));
    return;
  }

  if (req.mode === 'navigate') {
    event.respondWith(fetch(req).then(res => { caches.open(CACHE_NAME).then(cache => cache.put(req, res.clone())); return res; }).catch(() => caches.match(req).then(cached => cached || caches.match('/'))));
    return;
  }

  event.respondWith(caches.match(req).then(cached => cached || fetch(req).then(res => {
    if (res && res.type === 'basic') { const copy = res.clone(); caches.open(CACHE_NAME).then(cache => cache.put(req, copy)); }
    return res;
  }).catch(() => caches.match('/'))));
});
