// Minimal service worker: enables "Add to Home Screen" / "Install app" on
// Android and desktop Chrome/Edge. Caches the app shell only — live data
// (agents.json, run_log.jsonl, GitHub API calls) is always fetched fresh,
// never cached, since stale trading data would be actively misleading.

const CACHE_NAME = "agent-colony-shell-v1";
const SHELL_FILES = ["./index.html", "./manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls or live data files — always go to network.
  if (
    url.hostname.includes("github.com") ||
    url.hostname.includes("githubusercontent.com") ||
    url.pathname.includes("agents.json") ||
    url.pathname.includes("run_log.jsonl")
  ) {
    event.respondWith(fetch(event.request));
    return;
  }

  // App shell: cache-first for instant load, falls back to network.
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
