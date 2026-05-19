/**
 * frontend/src/components/WeatherWidget.js
 * Compact weather conditions widget component.
 *
 * Usage:
 *   const widget = new WeatherWidget(document.getElementById("weather-container"));
 *   widget.update({ temperature_c: 26, humidity_pct: 78, rainfall_mm: 3,
 *                   wind_speed_kmh: 12, visibility_km: 8, icon: "🌦️", description: "Rain showers" });
 */

export class WeatherWidget {
  constructor(container) {
    this.container = container;
    this._render();
  }

  _render() {
    this.container.innerHTML = `
      <div class="weather-widget">
        <div class="ww__icon" id="ww-icon">🌡️</div>
        <div class="ww__main">
          <span class="ww__temp" id="ww-temp">--°C</span>
          <span class="ww__desc" id="ww-desc">Loading weather...</span>
        </div>
        <div class="ww__grid">
          <div class="ww__stat"><span class="ww__stat-label">💧 Humidity</span><span class="ww__stat-val" id="ww-hum">--%</span></div>
          <div class="ww__stat"><span class="ww__stat-label">🌧️ Rain</span><span class="ww__stat-val" id="ww-rain">-- mm</span></div>
          <div class="ww__stat"><span class="ww__stat-label">💨 Wind</span><span class="ww__stat-val" id="ww-wind">-- km/h</span></div>
          <div class="ww__stat"><span class="ww__stat-label">👁️ Visibility</span><span class="ww__stat-val" id="ww-vis">-- km</span></div>
        </div>
      </div>
    `;

    this._el = {
      icon: this.container.querySelector("#ww-icon"),
      temp: this.container.querySelector("#ww-temp"),
      desc: this.container.querySelector("#ww-desc"),
      hum:  this.container.querySelector("#ww-hum"),
      rain: this.container.querySelector("#ww-rain"),
      wind: this.container.querySelector("#ww-wind"),
      vis:  this.container.querySelector("#ww-vis"),
    };
  }

  update(data) {
    this._el.icon.textContent = data.icon ?? "🌡️";
    this._el.temp.textContent = `${data.temperature_c ?? "--"}°C`;
    this._el.desc.textContent = data.description ?? "--";
    this._el.hum.textContent  = `${data.humidity_pct ?? "--"}%`;
    this._el.rain.textContent = `${data.rainfall_mm ?? "0"} mm`;
    this._el.wind.textContent = `${data.wind_speed_kmh ?? "--"} km/h`;
    this._el.vis.textContent  = `${data.visibility_km ?? "--"} km`;
  }
}
