(function () {
  "use strict";

  const cfg = window.APP_CONFIG;

  if (cfg.MODE !== "direct") {
    return;
  }

  const ui = window.PhysioUI;
  const prefix = cfg.TOPIC_PREFIX;

  const TOPIC = {
    data: prefix + "/data",
    actuator: prefix + "/actuator",
    cmd: prefix + "/cmd"
  };

  const MAX_ITEMS = 7200;
  const CSV_FIELDS = ["ts", "hr", "spo2", "temp", "hum", "room_temp", "finger", "status"];

  let client = null;
  let items = [];
  let lastDataAt = 0;
  let lastChartDraw = 0;

  let seq = Math.floor(Math.random() * 1000000000) + 1;

  function num(v) {
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  function handleData(msg) {
    const item = {
      ts: new Date().toISOString(),
      hr: num(msg.hr),
      spo2: num(msg.spo2),
      temp: num(msg.temp),
      hum: num(msg.hum),
      room_temp: num(msg.room_temp),
      finger: msg.finger === true,
      status: typeof msg.status === "string" ? msg.status : ""
    };

    items.push(item);

    if (items.length > MAX_ITEMS) {
      items.shift();
    }

    lastDataAt = Date.now();

    ui.renderLatest({ reading: item, prediction: null, local_status: item.status });

    if (Date.now() - lastChartDraw > 2000) {
      drawChart();
    }
  }

  function drawChart() {
    const since = Date.now() - cfg.HISTORY_MINUTES * 60 * 1000;

    lastChartDraw = Date.now();

    ui.renderHistory({
      items: items.filter((i) => new Date(i.ts).getTime() >= since)
    });
  }

  function onMessage(topic, payload) {
    let msg;

    try {
      msg = JSON.parse(payload.toString());
    } catch (e) {
      return;
    }

    if (topic === TOPIC.data) {
      handleData(msg);
    } else if (topic === TOPIC.actuator) {
      ui.renderActuator(msg);
    }
  }

  function connect() {
    client = mqtt.connect(cfg.MQTT_URL, {
      clientId: "dash-" + Math.random().toString(16).slice(2, 10),
      reconnectPeriod: 3000,
      connectTimeout: 10000,
      clean: true
    });

    client.on("connect", () => {
      client.subscribe([TOPIC.data, TOPIC.actuator]);
      ui.hideBanner();
    });

    client.on("close", () => {
      ui.showBanner("Disconnected from MQTT broker. Reconnecting.");
    });

    client.on("error", (e) => {
      ui.showBanner("MQTT error: " + e.message);
    });

    client.on("message", onMessage);
  }

  function publishCmd(body) {
    if (!client || !client.connected) {
      ui.showBanner("Not connected to MQTT broker.");
      return;
    }

    seq += 1;
    body.seq = seq;

    client.publish(TOPIC.cmd, JSON.stringify(body));
  }

  function exportCsv() {
    if (items.length === 0) {
      ui.showBanner("No data to download.");
      return;
    }

    const rows = items.map((i) =>
      CSV_FIELDS.map((k) => (i[k] === null || i[k] === undefined ? "" : i[k])).join(",")
    );

    const blob = new Blob([CSV_FIELDS.join(",") + "\n" + rows.join("\n")], { type: "text/csv" });
    const link = document.createElement("a");

    link.href = URL.createObjectURL(blob);
    link.download = "physio_" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".csv";
    link.click();

    URL.revokeObjectURL(link.href);
  }

  const directTransport = {
    start() {
      const note = document.getElementById("mode-note");

      note.textContent =
        "Direct mode (no backend): history only exists while the page is open, " +
        "status comes from firmware threshold and not ML model, and auto light " +
        "waits for model result. Use manual mode to test the light.";
      note.hidden = false;

      if (prefix.indexOf("GANTI-ID-UNIK") !== -1) {
        ui.showBanner("Fill TOPIC_PREFIX in js/config.js with the same value as in the firmware.");
        return;
      }

      connect();

      setInterval(() => {
        const age = lastDataAt === 0 ? null : (Date.now() - lastDataAt) / 1000;

        ui.renderDevice({ online: age !== null, age_s: age === null ? 9999 : age });
      }, 1000);
    },
    lampAuto() {
      publishCmd({ mode: "AUTO" });
    },
    lampManual(color, brightness) {
      publishCmd({ mode: "MANUAL", color: color, brightness: brightness });
    },
    buzzerTest(on) {
      publishCmd({ buzzer_test: on });
    },
    exportCsv: exportCsv
  };

  ui.useTransport(directTransport);
})();