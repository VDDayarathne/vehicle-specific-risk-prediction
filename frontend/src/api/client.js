/**
 * frontend/src/api/client.js
 * Thin fetch wrapper for communicating with the KaduGuard backend.
 */

const BASE_URL = window.KADUGUARD_API_URL || "http://localhost:8000";

async function apiFetch(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Public API methods ────────────────────────────────────────────────────────

export async function fetchWeather(lat, lon) {
  const params = lat && lon ? `?lat=${lat}&lon=${lon}` : "";
  return apiFetch(`/api/weather${params}`);
}

export async function predictRisk(payload) {
  return apiFetch("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchSegments() {
  return apiFetch("/api/segments");
}

export async function checkHealth() {
  return apiFetch("/api/health");
}
