/**
 * frontend/src/components/RiskCard.js
 * Reusable risk-level display card component (vanilla JS / DOM).
 *
 * Usage:
 *   const card = new RiskCard(document.getElementById("risk-container"));
 *   card.update({ risk_level: "High", risk_score: 0.87, vehicle_type: "bus",
 *                 recommended_speed_kmh: 20, alert_message: "..." });
 */

import { riskColor, riskEmoji, hexToRgbString } from "../utils/helpers.js";

export class RiskCard {
  /**
   * @param {HTMLElement} container  DOM node to render into.
   */
  constructor(container) {
    this.container = container;
    this._render();
  }

  _render() {
    this.container.innerHTML = `
      <div class="risk-card" id="risk-card-root">
        <div class="risk-card__header">
          <span class="risk-card__emoji" id="rc-emoji">⚪</span>
          <div class="risk-card__title-wrap">
            <span class="risk-card__label" id="rc-label">--</span>
            <span class="risk-card__sublabel">Risk Level</span>
          </div>
          <div class="risk-card__score-wrap">
            <span class="risk-card__score" id="rc-score">--%</span>
            <span class="risk-card__score-sub">Confidence</span>
          </div>
        </div>

        <div class="risk-card__bar-track">
          <div class="risk-card__bar-fill" id="rc-bar" style="width:0%"></div>
        </div>

        <div class="risk-card__details">
          <div class="risk-card__detail-item">
            <span class="risk-card__detail-icon">🚗</span>
            <span class="risk-card__detail-text" id="rc-vehicle">--</span>
          </div>
          <div class="risk-card__detail-item">
            <span class="risk-card__detail-icon">⚡</span>
            <span class="risk-card__detail-text" id="rc-speed">-- km/h</span>
          </div>
        </div>

        <p class="risk-card__message" id="rc-message">Awaiting prediction...</p>
      </div>
    `;

    this._el = {
      root:    this.container.querySelector("#risk-card-root"),
      emoji:   this.container.querySelector("#rc-emoji"),
      label:   this.container.querySelector("#rc-label"),
      score:   this.container.querySelector("#rc-score"),
      bar:     this.container.querySelector("#rc-bar"),
      vehicle: this.container.querySelector("#rc-vehicle"),
      speed:   this.container.querySelector("#rc-speed"),
      message: this.container.querySelector("#rc-message"),
    };
  }

  /**
   * Update the card with a fresh prediction response.
   * @param {{ risk_level, risk_score, vehicle_type, recommended_speed_kmh, alert_message }} data
   */
  update(data) {
    const { risk_level, risk_score, vehicle_type, recommended_speed_kmh, alert_message } = data;
    const color = riskColor(risk_level);
    const pct   = Math.round((risk_score ?? 0) * 100);

    this._el.root.style.setProperty("--rc-color", color);
    this._el.root.style.setProperty("--rc-rgb", hexToRgbString(color));
    this._el.emoji.textContent   = riskEmoji(risk_level);
    this._el.label.textContent   = risk_level ?? "--";
    this._el.score.textContent   = `${pct}%`;
    this._el.bar.style.width     = `${pct}%`;
    this._el.bar.style.background = color;
    this._el.vehicle.textContent = vehicle_type ?? "--";
    this._el.speed.textContent   = recommended_speed_kmh
      ? `Max ${recommended_speed_kmh} km/h`
      : "--";
    this._el.message.textContent = alert_message ?? "";

    // Pulse animation on update
    this._el.root.classList.remove("risk-card--pulse");
    void this._el.root.offsetWidth;  // reflow
    this._el.root.classList.add("risk-card--pulse");
  }

  /** Reset card to empty/loading state. */
  reset() {
    this._el.emoji.textContent   = "⚪";
    this._el.label.textContent   = "--";
    this._el.score.textContent   = "--%";
    this._el.bar.style.width     = "0%";
    this._el.vehicle.textContent = "--";
    this._el.speed.textContent   = "-- km/h";
    this._el.message.textContent = "Awaiting prediction...";
  }
}
