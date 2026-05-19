/**
 * frontend/src/pages/predict.js
 * Standalone prediction page logic — wires the form to the backend API.
 * Works with predict.html without any build step (native ES modules).
 */

import { fetchWeather, predictRisk } from "../api/client.js";
import { RiskCard }     from "../components/RiskCard.js";
import { WeatherWidget } from "../components/WeatherWidget.js";
import { formatTime }   from "../utils/helpers.js";

// ── DOM elements ──────────────────────────────────────────────────────────────

const form        = document.getElementById("predict-form");
const submitBtn   = document.getElementById("btn-submit");
const weatherTime = document.getElementById("weather-time");
const refreshBtn  = document.getElementById("btn-refresh-weather");
const alertWrap   = document.getElementById("prediction-alert");

const riskCard      = new RiskCard(document.getElementById("risk-card-container"));
const weatherWidget = new WeatherWidget(document.getElementById("weather-container"));

// ── Current weather state ─────────────────────────────────────────────────────

let currentWeather = {};
const DEVICE_ID_KEY = "kaduguard_device_id";
const DEVICE_ID = localStorage.getItem(DEVICE_ID_KEY) || (() => {
  const id = `device-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
  localStorage.setItem(DEVICE_ID_KEY, id);
  return id;
})();

let lastKnownPosition = null;

function renderAlert(result) {
  if (!alertWrap) return;
  const level = result.risk_level || "Low";
  const speed = result.recommended_speed_kmh;
  const cls = level.toLowerCase();
  const action = level === "High"
    ? "Stop or delay travel until conditions improve."
    : level === "Medium"
      ? "Reduce speed and keep extra distance on bends."
      : "Conditions are acceptable, but keep normal hill-road caution.";

  alertWrap.innerHTML = `
    <div class="prediction-alert prediction-alert--${cls}">
      <div class="prediction-alert__head">
        <span class="prediction-alert__pill">${level} Risk</span>
        <span class="prediction-alert__score">${Math.round((result.risk_score ?? 0) * 100)}%</span>
      </div>
      <div class="prediction-alert__msg">${result.alert_message ?? ""}</div>
      <div class="prediction-alert__meta">
        <span>Vehicle: <strong>${result.vehicle_type ?? "--"}</strong></span>
        <span>Recommended speed: <strong>${speed ?? "--"} km/h</strong></span>
      </div>
      <div class="prediction-alert__action">${action}</div>
    </div>
  `;
}

function attachGeolocation() {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      lastKnownPosition = {
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
      };
    },
    () => {
      lastKnownPosition = null;
    },
    { enableHighAccuracy: false, maximumAge: 60_000, timeout: 4_000 }
  );
}

async function loadWeather() {
  try {
    currentWeather = await fetchWeather();
    weatherWidget.update(currentWeather);
    weatherTime.textContent = `Updated ${formatTime()}`;

    // Auto-fill weather fields in the form
    document.getElementById("f-rainfall").value    = currentWeather.rainfall_mm   ?? 0;
    document.getElementById("f-humidity").value    = currentWeather.humidity_pct  ?? 70;
    document.getElementById("f-temp").value        = currentWeather.temperature_c ?? 25;
    document.getElementById("f-wind").value        = currentWeather.wind_speed_kmh ?? 10;
    document.getElementById("f-visibility").value  = currentWeather.visibility_km  ?? 10;
  } catch (err) {
    weatherTime.textContent = "⚠ Weather unavailable";
    console.warn("Weather fetch failed:", err);
  }
}

// ── Form submission ───────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = Object.fromEntries(new FormData(form));

  const payload = {
    device_id: DEVICE_ID,
    vehicle_type:     data.vehicle_type,
    temperature_c:    parseFloat(data.temperature_c)  || 25,
    humidity_pct:     parseFloat(data.humidity_pct)   || 70,
    rainfall_mm:      parseFloat(data.rainfall_mm)    || 0,
    wind_speed_kmh:   parseFloat(data.wind_speed_kmh) || 10,
    visibility_km:    parseFloat(data.visibility_km)  || 10,
    gradient_pct:     parseFloat(data.gradient_pct)   || 0,
    curvature:        parseFloat(data.curvature)       || 0,
    road_surface:     data.road_surface  || "asphalt",
    lane_count:       parseInt(data.lane_count, 10)   || 2,
    hour_of_day:      new Date().getHours(),
    day_of_week:      new Date().getDay() === 0 ? 6 : new Date().getDay() - 1,
    is_holiday:       data.is_holiday === "on",
    traffic_density:  parseFloat(data.traffic_density) || 0.5,
    ...(lastKnownPosition || {}),
  };

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="spinner"></span> Analysing...`;

  try {
    const result = await predictRisk(payload);
    riskCard.update(result);
    renderAlert(result);
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `🔍 Predict Risk`;
  }
});

// ── Refresh weather button ────────────────────────────────────────────────────

refreshBtn?.addEventListener("click", loadWeather);

// ── Init ──────────────────────────────────────────────────────────────────────

attachGeolocation();
loadWeather();
setInterval(loadWeather, 5 * 60 * 1000); // refresh every 5 min
