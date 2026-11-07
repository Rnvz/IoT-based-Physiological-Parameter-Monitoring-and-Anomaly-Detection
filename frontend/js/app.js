(function () {
  "use strict";

  const cfg = window.APP_CONFIG;
  const $ = (id) => document.getElementById(id);
  const KEY_STORE = "physio_api_key";

  const LABELS = {
    NORMAL: { text: "Normal", cls: "band-ok" },
    ANOMALY: { text: "Anomaly detected", cls: "band-bad" },
    NO_FINGER: { text: "Finger not detected", cls: "band-info" },
    CALCULATING: { text: "Calculating", cls: "band-info" },
    NO_DATA: { text: "No data", cls: "band-none" },
    SENSOR_ERROR: { text: "Sensor issue", cls: "band-warn" }
  };

  const COLOR_HEX = {
    GREEN: "#10b981",
    YELLOW: "#f59e0b",
    RED: "#ef4444",
    BLUE: "#3b82f6",
    PURPLE: "#a855f7",
    WHITE: "#94a3b8",
    OFF: "#333333"
  };

  const SOURCE_TEXT = {
    ML_RESULT: "model result",
    LOCAL_NO_FINGER: "finger not detected",
    NO_RESULT: "no model result yet",
    MANUAL: "dashboard command",
    NO_WIFI: "Wi-Fi not connected"
  };

  let selectedColor = "GREEN";
  const charts = {};
  let lampMode = "AUTO";
  let buzzerTestOn = false;
  let isAnomalyAlarm = false;
  let transport = null;

  function fmt(v, digits) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) {
      return "--";
    }
    return Number(v).toFixed(digits);
  }

  function validPositive(v) {
    return v !== null && v !== undefined && Number(v) > 0;
  }

  function getKey() {
    if (cfg.API_KEY) {
      return cfg.API_KEY;
    }
    const keyInput = $("api-key");
    if (keyInput && keyInput.value && keyInput.value.trim()) {
      saveKey(keyInput.value.trim());
      return keyInput.value.trim();
    }
    try {
      return sessionStorage.getItem(KEY_STORE) || "";
    } catch (e) {
      return "";
    }
  }

  function saveKey(value) {
    try {
      sessionStorage.setItem(KEY_STORE, value);
    } catch (e) {
      return;
    }
  }

  function showBanner(message) {
    const el = $("banner");
    el.textContent = message;
    el.hidden = false;
  }

  function hideBanner() {
    $("banner").hidden = true;
  }

  async function api(path, options) {
    const opts = options || {};
    const headers = { "Content-Type": "application/json" };
    const key = getKey();

    if (key) {
      headers["X-API-Key"] = key;
    }

    const res = await fetch(cfg.API_BASE + path, {
      method: opts.method || "GET",
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined
    });

    if (!res.ok) {
      throw new Error(path + " returned status " + res.status);
    }

    return res.json();
  }

  function setBand(labelKey, sub) {
    const info = LABELS[labelKey] || { text: "Waiting for data", cls: "band-none" };
    const band = $("status-band");

    band.className = "band " + info.cls;
    if (labelKey === "CALCULATING") {
      $("band-text").innerHTML = `${info.text}<span class="dots-anim"><span>.</span><span>.</span><span>.</span></span>`;
    } else {
      $("band-text").textContent = info.text;
    }

    if (sub) {
      $("band-sub").textContent = sub;
    }
  }

  function renderLocalBand(status) {
    const map = {
      NORMAL: { text: "Normal (local threshold)", cls: "band-ok" },
      ANOMALI: { text: "Anomaly (local threshold)", cls: "band-bad" },
      MENDETEKSI: { text: "Measuring", cls: "band-info" }
    };
    const info = map[status] || { text: "Waiting for data", cls: "band-none" };

    $("status-band").className = "band " + info.cls;
    if (status === "MENDETEKSI") {
      $("band-text").innerHTML = `${info.text}<span class="dots-anim"><span>.</span><span>.</span><span>.</span></span>`;
    } else {
      $("band-text").textContent = info.text;
    }
    $("band-sub").textContent = "Direct mode without backend: status comes from firmware threshold, not ML model.";
  }

  function renderLatest(data) {
    const reading = data && data.reading;
    const prediction = data && data.prediction;

    if (!reading) {
      setBand("NO_DATA");
      return;
    }

    const fingerOn = reading.finger === true;

    $("v-hr").textContent = fingerOn && validPositive(reading.hr) ? fmt(reading.hr, 0) : "--";
    $("v-spo2").textContent = fingerOn && validPositive(reading.spo2) ? fmt(reading.spo2, 0) : "--";
    $("v-temp").textContent = validPositive(reading.temp) ? fmt(reading.temp, 1) : "--";
    $("v-room").textContent = fmt(reading.room_temp, 1);
    $("v-hum").textContent = fmt(reading.hum, 0);

    if (reading.ts) {
      $("updated").textContent = "Last reading: " + new Date(reading.ts).toLocaleTimeString();
    }

    const activeScore = prediction ? (prediction.dl_score != null ? prediction.dl_score : prediction.score) : null;
    const activeMargin = prediction ? prediction.margin : null;
    const activeSource = prediction ? (prediction.decision_source || prediction.decision_mode || "--") : "--";
    const activeModel = prediction ? (prediction.model_version || "--") : "--";
    const activeLatency = prediction && prediction.inference_ms != null ? prediction.inference_ms + " ms" : "--";

    if ($("v-dl-score")) $("v-dl-score").textContent = activeScore != null ? Number(activeScore).toFixed(6) : "--";
    if ($("v-dl-margin")) $("v-dl-margin").textContent = activeMargin != null ? (activeMargin > 0 ? "+" + Number(activeMargin).toFixed(2) : Number(activeMargin).toFixed(2)) : "--";
    if ($("v-decision-source")) $("v-decision-source").textContent = activeSource;
    if ($("v-model-version")) $("v-model-version").textContent = activeModel;
    if ($("v-inference-ms")) $("v-inference-ms").textContent = activeLatency;

    const hasVitals = fingerOn && validPositive(reading.hr) && validPositive(reading.spo2);

    if (!fingerOn) {
      isAnomalyAlarm = false;
      setBand("NO_FINGER", "Place your finger on the MAX30102 sensor to start measuring.");
    } else if (!hasVitals) {

      isAnomalyAlarm = false;
      setBand("CALCULATING", "Detecting pulse and calculating heart rate...");
    } else if (data.local_status) {
      isAnomalyAlarm = false;
      renderLocalBand(data.local_status);
    } else if (prediction && LABELS[prediction.label]) {

      isAnomalyAlarm = prediction.label === "ANOMALY" || Number(prediction.severity) >= 2;
      const scoreStr = activeScore != null ? ` Anomaly score: ${Number(activeScore).toFixed(6)} (margin: ${activeMargin != null ? (activeMargin > 0 ? "+" + Number(activeMargin).toFixed(2) : Number(activeMargin).toFixed(2)) : "--"}).` : "";
      setBand(prediction.label, "Model result indicates unusual data patterns, not a medical diagnosis." + scoreStr);
    } else {
      isAnomalyAlarm = false;
      setBand("CALCULATING", "Collecting sensor data samples for model evaluation...");
    }
    updateBuzzerIndicator();
  }

  function renderDevice(data) {
    const pill = $("device-pill");
    const online = data && data.online === true && Number(data.age_s) <= cfg.OFFLINE_AFTER_S;

    pill.textContent = online ? "Device: online" : "Device: offline";
    pill.className = "pill " + (online ? "pill-on" : "pill-off");
  }

  function renderActuator(data) {
    if (!data || !data.led) {
      return;
    }

    lampMode = data.mode === "MANUAL" ? "MANUAL" : "AUTO";

    const color = data.led.color || "OFF";
    const brightness = Number(data.led.brightness) || 0;
    const swatch = $("lamp-swatch");

    swatch.style.background = COLOR_HEX[color] || COLOR_HEX.OFF;
    swatch.style.opacity = String(0.3 + 0.7 * (brightness / 100));

    $("lamp-state").textContent = lampMode === "AUTO" ? "Auto mode" : "Manual mode";

    let detail = "Color " + color.toLowerCase() + ", brightness " + brightness + "%";
    if (data.source) {
      detail += ", source: " + (SOURCE_TEXT[data.source] || data.source);
    }
    if (lampMode === "MANUAL" && data.manual_left_s) {
      detail += ", returns to auto in " + Math.ceil(data.manual_left_s / 60) + " minutes";
    }
    $("lamp-detail").textContent = detail;

    $("btn-auto").setAttribute("aria-pressed", String(lampMode === "AUTO"));
    $("btn-manual").setAttribute("aria-pressed", String(lampMode === "MANUAL"));

    buzzerTestOn = !!(data.buzzer && data.buzzer.test);
    const bb = $("btn-buzzer");
    bb.setAttribute("aria-pressed", String(buzzerTestOn));
    bb.textContent = buzzerTestOn ? "Buzzer: on (click to turn off)" : "Buzzer: off";

    const indLamp = $("ind-lamp");
    const indLampText = $("ind-lamp-text");

    if (color === "OFF" || brightness === 0) {
      indLamp.style.background = "transparent";
      indLamp.style.borderColor = "var(--line)";
      indLamp.style.boxShadow = "none";
      indLampText.textContent = "OFF";
    } else {
      const hexColor = COLOR_HEX[color] || COLOR_HEX.OFF;
      indLamp.style.background = hexColor;
      indLamp.style.borderColor = hexColor;

      const glowSpread = (brightness / 100) * 12;
      indLamp.style.boxShadow = `0 0 ${glowSpread}px ${glowSpread / 2}px ${hexColor}`;
      indLampText.textContent = color;
    }

    updateBuzzerIndicator();
  }

  function updateBuzzerIndicator() {
    const isBuzzerActive = buzzerTestOn || isAnomalyAlarm;
    const indBuzzer = $("ind-buzzer");
    const indBuzzerText = $("ind-buzzer-text");

    if (!indBuzzer || !indBuzzerText) return;

    if (isBuzzerActive) {
      indBuzzer.className = "ind-circle buzzer-on";
      indBuzzerText.textContent = "ON";
      indBuzzerText.style.color = "var(--bad)";
    } else {
      indBuzzer.className = "ind-circle buzzer-off";
      indBuzzerText.textContent = "OFF";
      indBuzzerText.style.color = "var(--ink)";
    }
  }

  function createChart(ctxId, label, lineColor, bgColor, sMin, sMax) {
    return new Chart($(ctxId), {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label: label,
          data: [],
          borderColor: lineColor,
          backgroundColor: bgColor,
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 0,
          pointHitRadius: 15,
          fill: true
        }]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            ticks: { maxTicksLimit: 8, font: { size: 10 } },
            grid: { display: false }
          },
          y: {
            suggestedMin: sMin,
            suggestedMax: sMax,
            ticks: { font: { size: 10 } },
            grid: { color: "rgba(150, 150, 150, 0.15)" }
          }
        }
      }
    });
  }

  function initCharts() {
    charts.hr = createChart("chart-hr", "Heart Rate", "#ef4444", "rgba(239, 68, 68, 0.15)", 50, 150);
    charts.spo2 = createChart("chart-spo2", "SpO2", "#3b82f6", "rgba(59, 130, 246, 0.15)", 85, 100);
    charts.temp = createChart("chart-temp", "Skin Temp", "#f59e0b", "rgba(245, 158, 11, 0.15)", 30, 40);
    charts.room = createChart("chart-room", "Room Temp", "#8b5cf6", "rgba(139, 92, 246, 0.15)", 20, 35);
    charts.hum = createChart("chart-hum", "Humidity", "#10b981", "rgba(16, 185, 129, 0.15)", 40, 80);

    const scoreCanvas = $("chart-score");
    if (scoreCanvas) {
      charts.score = new Chart(scoreCanvas, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Anomaly Score",
              data: [],
              borderColor: "#8b5cf6",
              backgroundColor: "rgba(139, 92, 246, 0.15)",
              borderWidth: 2,
              tension: 0.2,
              pointRadius: 2,
              fill: false
            },
            {
              label: "Warn Thresh (+0.080050)",
              data: [],
              borderColor: "#f59e0b",
              borderDash: [5, 5],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            },
            {
              label: "Crit Thresh (-0.084044)",
              data: [],
              borderColor: "#ef4444",
              borderDash: [5, 5],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              labels: { font: { size: 10 }, boxWidth: 12 }
            }
          },
          scales: {
            x: {
              ticks: { maxTicksLimit: 8, font: { size: 10 } },
              grid: { display: false }
            },
            y: {
              suggestedMin: -0.20,
              suggestedMax: 0.20,
              ticks: { font: { size: 10 } },
              grid: { color: "rgba(150, 150, 150, 0.15)" }
            }
          }
        }
      });
    }
  }

  function renderPredictions(data) {
    const items = (data && data.items) || [];
    if (!items.length) {
      return;
    }

    const sorted = [...items].sort((a, b) => new Date(a.ts) - new Date(b.ts));
    const labels = sorted.map((p) => new Date(p.ts).toLocaleTimeString());

    const lastItem = sorted[sorted.length - 1];
    const isDL = lastItem && (lastItem.dl_score != null || (lastItem.model_version && lastItem.model_version.includes("ae")));

    const scoreTitle = $("score-chart-title");
    const threshPill = $("score-threshold-pill");

    let warnThreshVal = 0.080050;
    let critThreshVal = -0.084044;
    let warnLabel = "Warn Thresh (+0.080050)";
    let critLabel = "Crit Thresh (-0.084044)";
    let sMin = -0.20;
    let sMax = 0.20;

    if (isDL) {
      warnThreshVal = 0.006000;
      critThreshVal = 0.010400;
      warnLabel = "Warn Thresh (0.006000)";
      critLabel = "Crit Thresh (0.010400)";
      sMin = 0.0;
      sMax = 0.035;
      if (scoreTitle) scoreTitle.textContent = "DL Anomaly Score (Dense Autoencoder)";
      if (threshPill) threshPill.textContent = "Warn: 0.006000 | Crit: 0.010400";
    } else {
      if (scoreTitle) scoreTitle.textContent = "ML Anomaly Score (Isolation Forest)";
      if (threshPill) threshPill.textContent = "Warn: +0.080050 | Crit: -0.084044";
    }

    const scoreData = sorted.map((p) => (isDL && p.dl_score != null ? p.dl_score : (p.score != null ? p.score : p.dl_score)));
    const warnData = sorted.map(() => warnThreshVal);
    const critData = sorted.map(() => critThreshVal);

    if (charts.score) {
      charts.score.options.scales.y.suggestedMin = sMin;
      charts.score.options.scales.y.suggestedMax = sMax;
      charts.score.data.labels = labels;
      charts.score.data.datasets[0].data = scoreData;
      charts.score.data.datasets[1].label = warnLabel;
      charts.score.data.datasets[1].data = warnData;
      charts.score.data.datasets[2].label = critLabel;
      charts.score.data.datasets[2].data = critData;
      charts.score.update("none");
    }

    const tbody = $("timeline-tbody");
    if (!tbody) return;

    const recent = [...items].reverse().slice(0, 15);
    if (!recent.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-center muted">Belum ada riwayat keputusan</td></tr>';
      return;
    }

    let rowsHtml = "";
    recent.forEach((p) => {
      const timeStr = new Date(p.ts).toLocaleTimeString();
      const ruleSevBadge = p.rule_severity != null
        ? `<span class="sev-badge sev-${p.rule_severity}">Sev ${p.rule_severity}</span>`
        : `<span class="muted">-</span>`;

      const dlVal = p.dl_severity != null ? p.dl_severity : p.ml_severity;
      const dlSevBadge = dlVal != null
        ? `<span class="sev-badge sev-${dlVal}">Sev ${dlVal}</span>`
        : `<span class="muted">- (off)</span>`;

      const pubSevBadge = `<span class="sev-badge sev-${p.severity}">Sev ${p.severity}</span>`;
      const sourceStr = p.decision_source || p.decision_mode || "--";
      const rawScoreStr = p.dl_score != null ? Number(p.dl_score).toFixed(6) : (p.score != null ? Number(p.score).toFixed(4) : "-");
      const marginStr = p.margin != null ? (p.margin > 0 ? "+" + Number(p.margin).toFixed(2) : Number(p.margin).toFixed(2)) : "-";
      const statusLabel = p.label || "NORMAL";

      rowsHtml += `<tr>
        <td>${timeStr}</td>
        <td>${ruleSevBadge}</td>
        <td>${dlSevBadge}</td>
        <td><strong>${sourceStr}</strong></td>
        <td>${pubSevBadge}</td>
        <td><code>${rawScoreStr}</code></td>
        <td>${marginStr}</td>
        <td>${statusLabel}</td>
      </tr>`;
    });
    tbody.innerHTML = rowsHtml;
  }

  function renderHistory(data) {
    const items = (data && data.items) || [];
    const labels = items.map((i) => new Date(i.ts).toLocaleTimeString());

    const hrData = items.map((i) => (i.finger !== false && validPositive(i.hr) ? i.hr : null));
    const spo2Data = items.map((i) => (i.finger !== false && validPositive(i.spo2) ? i.spo2 : null));
    const tempData = items.map((i) => (validPositive(i.temp) ? i.temp : null));
    const roomData = items.map((i) => (validPositive(i.room_temp) ? i.room_temp : null));
    const humData = items.map((i) => (validPositive(i.hum) ? i.hum : null));

    function update(c, lab, val) {
      if (!c) return;
      c.data.labels = lab;
      c.data.datasets[0].data = val;
      c.update("none");
    }

    update(charts.hr, labels, hrData);
    update(charts.spo2, labels, spo2Data);
    update(charts.temp, labels, tempData);
    update(charts.room, labels, roomData);
    update(charts.hum, labels, humData);
  }

  async function refreshLive() {
    const results = await Promise.allSettled([
      api("/latest"),
      api("/device/status"),
      api("/actuator/status")
    ]);

    if (results[0].status === "fulfilled") {
      renderLatest(results[0].value);
      hideBanner();
    } else {
      showBanner("Cannot connect to API at " + cfg.API_BASE + ". Check backend and CORS settings.");
    }

    if (results[1].status === "fulfilled") {
      renderDevice(results[1].value);
    }

    if (results[2].status === "fulfilled") {
      renderActuator(results[2].value);
    }
  }

  async function refreshHistory() {
    try {
      renderHistory(await api("/history?minutes=" + cfg.HISTORY_MINUTES));
    } catch (e) {

    }
    try {
      renderPredictions(await api("/predictions?limit=30"));
    } catch (e) {

    }
  }

  async function sendAction(path, body) {
    if (!getKey()) {
      showBanner("Fill in the API key at the top to control the light/buzzer.");
      return;
    }

    try {
      await api(path, { method: "POST", body: body });
      hideBanner();
      renderActuator(await api("/actuator/status"));
    } catch (e) {
      showBanner("Failed to send command: " + e.message);
    }
  }

  function sendManualLed() {
    transport.lampManual(selectedColor, Number($("brightness").value));
  }

  function markSelectedColor() {
    document.querySelectorAll(".colors button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.color === selectedColor));
    });
  }

  function bindControls() {
    const keyInput = $("api-key");
    if (keyInput) {
      keyInput.value = getKey();
      keyInput.addEventListener("input", () => saveKey(keyInput.value.trim()));
      keyInput.addEventListener("change", () => saveKey(keyInput.value.trim()));
    }

    $("btn-auto").addEventListener("click", () => transport.lampAuto());
    $("btn-buzzer").addEventListener("click", () => transport.buzzerTest(!buzzerTestOn));
    $("btn-csv").addEventListener("click", () => transport.exportCsv());
    $("btn-manual").addEventListener("click", sendManualLed);

    document.querySelectorAll(".colors button").forEach((b) => {
      b.addEventListener("click", () => {
        selectedColor = b.dataset.color;
        markSelectedColor();
        sendManualLed();
      });
    });

    const slider = $("brightness");

    slider.addEventListener("input", () => {
      $("bright-out").textContent = slider.value;
    });

    slider.addEventListener("change", () => {
      if (lampMode === "MANUAL") {
        sendManualLed();
      }
    });

    markSelectedColor();
  }

  const apiTransport = {
    start() {
      refreshLive();
      refreshHistory();

      setInterval(refreshLive, cfg.POLL_MS);
      setInterval(refreshHistory, 15000);
    },
    lampAuto() {
      sendAction("/actuator/mode", { mode: "AUTO" });
    },
    lampManual(color, brightness) {
      sendAction("/actuator/led", { color: color, brightness: brightness });
    },
    buzzerTest(on) {
      sendAction("/actuator/buzzer", { test: on });
    },
    async exportCsv() {
      const key = getKey();
      if (!key) {
        showBanner("Enter the API key in the top bar to download the dataset CSV.");
        return;
      }

      showBanner("Preparing dataset export...");
      try {
        const res = await fetch(cfg.API_BASE + "/export/dataset.csv", {
          method: "GET",
          headers: {
            "X-API-Key": key
          }
        });

        if (res.status === 401) {
          showBanner("Failed to export dataset: Invalid or missing API key.");
          return;
        }

        if (!res.ok) {
          showBanner("Failed to export dataset: server returned status " + res.status);
          return;
        }

        const blob = await res.blob();
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "dataset_" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".csv";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);

        hideBanner();
      } catch (e) {
        showBanner("Failed to download dataset CSV: " + e.message);
      }
    }
  };

  function useTransport(t) {
    transport = t;
    t.start();
  }

  window.PhysioUI = {
    renderLatest: renderLatest,
    renderDevice: renderDevice,
    renderActuator: renderActuator,
    renderHistory: renderHistory,
    showBanner: showBanner,
    hideBanner: hideBanner,
    useTransport: useTransport
  };

  initCharts();
  bindControls();

  if (cfg.MODE === "direct") {
    document.querySelector(".key").hidden = true;
    $("btn-csv").hidden = false;
  } else {
    $("btn-csv").hidden = false;
    useTransport(apiTransport);
  }
})();
