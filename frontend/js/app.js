(function () {
  "use strict";

  const cfg = window.APP_CONFIG;
  const $ = (id) => document.getElementById(id);
  const KEY_STORE = "physio_api_key";

  const LABELS = {
    NORMAL: { text: "Normal", cls: "band-ok" },
    ANOMALY: { text: "Anomaly detected", cls: "band-bad" },
    NO_FINGER: { text: "Finger not detected", cls: "band-info" },
    NO_DATA: { text: "No data", cls: "band-none" },
    SENSOR_ERROR: { text: "Sensor issue", cls: "band-warn" }
  };

  const COLOR_HEX = {
    GREEN: "#10b981",
    YELLOW: "#f59e0b",
    RED: "#ef4444",
    BLUE: "#3b82f6",
    WHITE: "#94a3b8",
    OFF: "#333333"
  };

  const SOURCE_TEXT = {
    ML_RESULT: "model result",
    LOCAL_NO_FINGER: "finger not detected",
    NO_RESULT: "no model result yet",
    MANUAL: "dashboard command"
  };

  let selectedColor = "GREEN";
  const charts = {};
  let lampMode = "AUTO";
  let buzzerOn = false;
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
    $("band-text").textContent = info.text;

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
    $("band-text").textContent = info.text;
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

    if (!fingerOn) {
      setBand("NO_FINGER", "Place your finger on the MAX30102 sensor to start measuring.");
    } else if (data.local_status) {
      renderLocalBand(data.local_status);
    } else if (prediction && LABELS[prediction.label]) {
      const score = prediction.score === null || prediction.score === undefined
        ? ""
        : " Anomaly score: " + fmt(prediction.score, 3) + ".";
      setBand(prediction.label, "Model result indicates unusual data patterns, not a medical diagnosis." + score);
    } else {
      setBand("NO_DATA", "Waiting for model result.");
    }
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

    buzzerOn = !!(data.buzzer && data.buzzer.test);
    const bb = $("btn-buzzer");
    bb.setAttribute("aria-pressed", String(buzzerOn));
    bb.textContent = buzzerOn ? "Buzzer: on (click to turn off)" : "Buzzer: off";

    /* --- UPDATE INDIKATOR CARD BARU --- */
    
    // 1. Indikator Lampu (Glow effect berdasarkan brightness)
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
      // Efek cahaya (glow) menyesuaikan tingkat kecerahan
      const glowSpread = (brightness / 100) * 12;
      indLamp.style.boxShadow = `0 0 ${glowSpread}px ${glowSpread / 2}px ${hexColor}`;
      indLampText.textContent = color;
    }

    // 2. Indikator Buzzer (Pulse animation jika menyala)
    const indBuzzer = $("ind-buzzer");
    const indBuzzerText = $("ind-buzzer-text");

    if (buzzerOn) {
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
      return;
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

    keyInput.value = getKey();
    keyInput.addEventListener("input", () => saveKey(keyInput.value.trim()));
    keyInput.addEventListener("change", () => saveKey(keyInput.value.trim()));

    $("btn-auto").addEventListener("click", () => transport.lampAuto());
    $("btn-buzzer").addEventListener("click", () => transport.buzzerTest(!buzzerOn));
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
    exportCsv() {
      showBanner("CSV download is only available in direct mode.");
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
    useTransport(apiTransport);
  }
})();