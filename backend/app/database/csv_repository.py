import os
import csv
import json
import bisect
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Set, Tuple, Dict, Any, Generator
from app.database.interfaces import (
    BaseRepository,
    ReadingRecord,
    PredictionRecord,
    ActuatorEventRecord
)

class CSVRepository(BaseRepository):
    """
    CSV File-based implementation of BaseRepository.
    Stores logs in:
      - readings.csv
      - predictions.csv
      - actuator_events.csv
    Handles deduplication on (device_id, ts).
    """

    def __init__(self, csv_dir: str = "./data"):
        self.csv_dir = os.path.abspath(csv_dir)
        os.makedirs(self.csv_dir, exist_ok=True)

        self.readings_file = os.path.join(self.csv_dir, "readings.csv")
        self.predictions_file = os.path.join(self.csv_dir, "predictions.csv")
        self.actuator_file = os.path.join(self.csv_dir, "actuator_events.csv")

        self._lock = asyncio.Lock()
        self._seen_readings: Set[Tuple[str, str]] = set()
        self._seen_predictions: Set[Tuple[str, str]] = set()

        self._latest_readings: dict = {}
        self._latest_predictions: dict = {}
        self._latest_actuator: Optional[ActuatorEventRecord] = None

        self._init_files()

    def _init_files(self):
        # 1. readings.csv
        if not os.path.exists(self.readings_file):
            with open(self.readings_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "device_id", "hr", "spo2", "temp", "hum", "room_temp", "finger", "fw_status"])
        else:
            # Seed seen set to prevent duplicate appends on restart
            try:
                with open(self.readings_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dev = row.get("device_id")
                        ts = row.get("ts")
                        if dev and ts:
                            self._seen_readings.add((dev, ts))
            except Exception:
                pass

        # 2. predictions.csv
        if not os.path.exists(self.predictions_file):
            with open(self.predictions_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ts", "device_id", "rule_severity", "ml_severity", "ml_score",
                    "published_severity", "decision_mode", "decision_source", "label", "model_version"
                ])
        else:
            try:
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dev = row.get("device_id")
                        ts = row.get("ts")
                        if dev and ts:
                            self._seen_predictions.add((dev, ts))
            except Exception:
                pass

        # 3. actuator_events.csv
        if not os.path.exists(self.actuator_file):
            with open(self.actuator_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "kind", "seq", "payload_json"])

    async def save_reading(self, reading: ReadingRecord) -> bool:
        key = (reading.device_id, reading.ts)
        async with self._lock:
            if key in self._seen_readings:
                return False  # Duplicate ignored

            self._seen_readings.add(key)
            self._latest_readings[reading.device_id] = reading

            try:
                with open(self.readings_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        reading.ts,
                        reading.device_id,
                        "" if reading.hr is None else reading.hr,
                        "" if reading.spo2 is None else reading.spo2,
                        "" if reading.temp is None else reading.temp,
                        "" if reading.hum is None else reading.hum,
                        "" if reading.room_temp is None else reading.room_temp,
                        reading.finger,
                        reading.fw_status or "NORMAL"
                    ])
                return True
            except Exception as e:
                # Log error without propagating crash
                print(f"[CSVRepository] Error writing reading: {e}")
                return False

    async def get_latest_reading(self, device_id: str) -> Optional[ReadingRecord]:
        async with self._lock:
            if device_id in self._latest_readings:
                return self._latest_readings[device_id]
        
        # Fallback reading backwards with deque
        try:
            with open(self.readings_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                matched = [row for row in reader if row.get("device_id") == device_id]
                if matched:
                    rec = self._row_to_reading(matched[-1])
                    async with self._lock:
                        self._latest_readings[device_id] = rec
                    return rec
        except Exception:
            pass
        return None

    async def get_latest_prediction(self, device_id: str) -> Optional[PredictionRecord]:
        async with self._lock:
            if device_id in self._latest_predictions:
                return self._latest_predictions[device_id]

        try:
            with open(self.predictions_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                matched = [row for row in reader if row.get("device_id") == device_id]
                if matched:
                    rec = self._row_to_prediction(matched[-1])
                    async with self._lock:
                        self._latest_predictions[device_id] = rec
                    return rec
        except Exception:
            pass
        return None

    async def get_history(self, device_id: str, minutes: int = 30) -> List[ReadingRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        results = []
        try:
            with open(self.readings_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("device_id") != device_id:
                        continue
                    ts_str = row.get("ts")
                    if not ts_str:
                        continue
                    try:
                        row_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if row_dt >= cutoff:
                            results.append(self._row_to_reading(row))
                    except Exception:
                        results.append(self._row_to_reading(row))
        except Exception as e:
            print(f"[CSVRepository] Error reading history: {e}")
        return results

    async def save_prediction(self, prediction: PredictionRecord) -> bool:
        key = (prediction.device_id, prediction.ts)
        async with self._lock:
            if key in self._seen_predictions:
                return False

            self._seen_predictions.add(key)
            self._latest_predictions[prediction.device_id] = prediction

            try:
                with open(self.predictions_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        prediction.ts,
                        prediction.device_id,
                        "" if prediction.rule_severity is None else prediction.rule_severity,
                        "" if prediction.ml_severity is None else prediction.ml_severity,
                        "" if prediction.ml_score is None else f"{prediction.ml_score:.4f}",
                        prediction.published_severity,
                        prediction.decision_mode,
                        prediction.decision_source,
                        prediction.label,
                        prediction.model_version
                    ])
                return True
            except Exception as e:
                print(f"[CSVRepository] Error writing prediction: {e}")
                return False

    async def get_predictions(self, device_id: str, limit: int = 100) -> List[PredictionRecord]:
        from collections import deque
        results = deque(maxlen=limit)
        try:
            with open(self.predictions_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("device_id") == device_id:
                        results.append(self._row_to_prediction(row))
        except Exception as e:
            print(f"[CSVRepository] Error reading predictions: {e}")
        return list(results)

    async def save_actuator_event(self, event: ActuatorEventRecord) -> bool:
        async with self._lock:
            self._latest_actuator = event
            try:
                with open(self.actuator_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        event.ts,
                        event.kind,
                        "" if event.seq is None else event.seq,
                        event.payload_json
                    ])
                return True
            except Exception as e:
                print(f"[CSVRepository] Error writing actuator event: {e}")
                return False

    async def get_latest_actuator_event(self, kind: Optional[str] = None) -> Optional[ActuatorEventRecord]:
        async with self._lock:
            if self._latest_actuator:
                if kind is None or self._latest_actuator.kind == kind:
                    return self._latest_actuator

        try:
            with open(self.actuator_file, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                for row in reversed(rows):
                    if kind is None or row.get("kind") == kind:
                        return ActuatorEventRecord(
                            ts=row.get("ts", ""),
                            kind=row.get("kind", ""),
                            seq=int(row["seq"]) if row.get("seq") else None,
                            payload_json=row.get("payload_json", "{}")
                        )
        except Exception:
            pass
        return None

    def _row_to_reading(self, row: dict) -> ReadingRecord:
        def parse_float(val):
            if val is None or val == "":
                return None
            try:
                return float(val)
            except ValueError:
                return None

        finger_raw = row.get("finger", "False")
        finger_bool = finger_raw.lower() in ("true", "1", "t")

        return ReadingRecord(
            ts=row.get("ts", ""),
            device_id=row.get("device_id", "esp32-01"),
            hr=parse_float(row.get("hr")),
            spo2=parse_float(row.get("spo2")),
            temp=parse_float(row.get("temp")),
            hum=parse_float(row.get("hum")),
            room_temp=parse_float(row.get("room_temp")),
            finger=finger_bool,
            fw_status=row.get("fw_status", "NORMAL")
        )

    def _row_to_prediction(self, row: dict) -> PredictionRecord:
        def parse_int(val):
            if val is None or val == "":
                return None
            try:
                return int(val)
            except ValueError:
                return None

        def parse_float(val):
            if val is None or val == "":
                return None
            try:
                return float(val)
            except ValueError:
                return None

        return PredictionRecord(
            ts=row.get("ts", ""),
            device_id=row.get("device_id", "esp32-01"),
            rule_severity=parse_int(row.get("rule_severity")),
            ml_severity=parse_int(row.get("ml_severity")),
            ml_score=parse_float(row.get("ml_score")),
            published_severity=int(row.get("published_severity", 0)),
            decision_mode=row.get("decision_mode", "RULE"),
            decision_source=row.get("decision_source", row.get("decision_mode", "RULE")),
            label=row.get("label", "NORMAL"),
            model_version=row.get("model_version", "if_v1")
        )

    def stream_dataset_export(self) -> Generator[str, None, None]:
        """
        Streams one row per reading joined with prediction and last known actuator state as of reading ts.
        Header:
        ts,hr,spo2,temp,hum,room_temp,finger,fw_status,rule_severity,ml_severity,published_severity,decision_mode,mode,led_color,brightness,buzzer_test,source
        """
        # 1. Load predictions lookup
        pred_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if os.path.exists(self.predictions_file):
            try:
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dev = row.get("device_id", "")
                        ts = row.get("ts", "")
                        pred_lookup[(dev, ts)] = row
            except Exception as e:
                print(f"[CSVRepository] Error reading predictions for export: {e}")

        # 2. Load actuator events timeline
        actuator_timeline: List[Tuple[str, Dict[str, Any]]] = []
        if os.path.exists(self.actuator_file):
            try:
                with open(self.actuator_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = row.get("ts", "")
                        parsed = parse_actuator_payload(row.get("payload_json", "{}"))
                        actuator_timeline.append((ts, parsed))
            except Exception as e:
                print(f"[CSVRepository] Error reading actuator events for export: {e}")

        actuator_timeline.sort(key=lambda x: x[0])
        actuator_ts_list = [x[0] for x in actuator_timeline]
        default_actuator = {
            "mode": "AUTO",
            "led_color": "OFF",
            "brightness": 100,
            "buzzer_test": False,
            "source": "BOOT"
        }

        # 3. Stream output CSV
        header = [
            "ts", "hr", "spo2", "temp", "hum", "room_temp", "finger", "fw_status",
            "rule_severity", "ml_severity", "published_severity", "decision_mode",
            "mode", "led_color", "brightness", "buzzer_test", "source"
        ]
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        yield buf.getvalue()

        if not os.path.exists(self.readings_file):
            return

        with open(self.readings_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get("ts", "")
                dev = row.get("device_id", "")
                pred = pred_lookup.get((dev, ts), {})
                act = as_of_join_actuator(ts, actuator_timeline, actuator_ts_list, default_actuator)

                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    ts,
                    row.get("hr", ""),
                    row.get("spo2", ""),
                    row.get("temp", ""),
                    row.get("hum", ""),
                    row.get("room_temp", ""),
                    row.get("finger", "False"),
                    row.get("fw_status", "NORMAL"),
                    pred.get("rule_severity", ""),
                    pred.get("ml_severity", ""),
                    pred.get("published_severity", ""),
                    pred.get("decision_mode", ""),
                    act["mode"],
                    act["led_color"],
                    act["brightness"],
                    act["buzzer_test"],
                    act["source"]
                ])
                yield buf.getvalue()


def parse_actuator_payload(payload_json: str) -> Dict[str, Any]:
    """Parse actuator payload handling both flat and nested JSON schemas."""
    try:
        data = json.loads(payload_json)
    except Exception:
        data = {}

    mode = data.get("mode", "AUTO")

    # Handle nested {"led": {"color": "...", "brightness": ...}} or flat keys
    if isinstance(data.get("led"), dict):
        led_color = data["led"].get("color", "OFF")
        brightness = data["led"].get("brightness", 100)
    else:
        led_color = data.get("led_color", "OFF")
        brightness = data.get("brightness", 100)

    # Handle nested {"buzzer": {"test": bool}} or flat keys
    if isinstance(data.get("buzzer"), dict):
        buzzer_test = data["buzzer"].get("test", False)
    else:
        buzzer_test = data.get("buzzer_test", False)

    source = data.get("source", "BOOT")

    return {
        "mode": mode,
        "led_color": led_color,
        "brightness": brightness,
        "buzzer_test": buzzer_test,
        "source": source
    }


def as_of_join_actuator(
    reading_ts: str,
    actuator_timeline: List[Tuple[str, Dict[str, Any]]],
    actuator_ts_list: List[str],
    default_actuator: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Finds the latest actuator state whose timestamp <= reading_ts.
    If no prior actuator event exists, returns default_actuator.
    """
    if default_actuator is None:
        default_actuator = {
            "mode": "AUTO",
            "led_color": "OFF",
            "brightness": 100,
            "buzzer_test": False,
            "source": "BOOT"
        }
    idx = bisect.bisect_right(actuator_ts_list, reading_ts) - 1
    if idx >= 0:
        return actuator_timeline[idx][1]
    return default_actuator

