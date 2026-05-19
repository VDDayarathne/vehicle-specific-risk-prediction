/**
 * frontend/src/utils/helpers.js
 * Shared utility functions used across components.
 */

/**
 * Return colour hex for a risk label.
 * @param {"Low"|"Medium"|"High"} label
 */
export function riskColor(label) {
  const map = { Low: "#2ed573", Medium: "#ff8c42", High: "#ff4757" };
  return map[label] ?? "#8ba4c0";
}

/**
 * Convert a hex color to an "r,g,b" string.
 */
export function hexToRgbString(hex) {
  const clean = hex.replace("#", "").trim();
  const full = clean.length === 3
    ? clean.split("").map(ch => ch + ch).join("")
    : clean;
  const num = parseInt(full, 16);
  if (Number.isNaN(num) || full.length !== 6) return "255,255,255";
  return `${(num >> 16) & 255},${(num >> 8) & 255},${num & 255}`;
}

/**
 * Return an emoji badge for a risk label.
 */
export function riskEmoji(label) {
  return { Low: "🟢", Medium: "🟡", High: "🔴" }[label] ?? "⚪";
}

/**
 * Return speed recommendation per vehicle type and risk.
 */
export const SPEED_MAP = {
  Low:    { car: 60, motorcycle: 50, bus: 50, lorry: 45, "three-wheeler": 40 },
  Medium: { car: 40, motorcycle: 30, bus: 35, lorry: 30, "three-wheeler": 25 },
  High:   { car: 20, motorcycle: 15, bus: 20, lorry: 15, "three-wheeler": 10 },
};

/**
 * Format a Date object as "HH:MM".
 */
export function formatTime(date = new Date()) {
  return date.toLocaleTimeString("en-LK", { hour: "2-digit", minute: "2-digit" });
}

/**
 * Debounce a function.
 */
export function debounce(fn, ms = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/**
 * Clamp a value between min and max.
 */
export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}
