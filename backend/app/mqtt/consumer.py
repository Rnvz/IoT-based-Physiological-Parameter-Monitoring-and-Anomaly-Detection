import json
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable
import paho.mqtt.client as mqtt
from pydantic import ValidationError

from app.config import settings
from app.schemas.telemetry import TelemetryPayload
from app.schemas.result import ResultPayload
from app.database.interfaces import ReadingRecord, PredictionRecord, ActuatorEventRecord
from app.database.factory import get_repository
from app.rules.engine import evaluate_rules
from app.rules.arbitrator import arbitrate
from app.ml.inference import get_ml_service
from app.mqtt.publisher import MQTTPublisher

logger = logging.getLogger(__name__)

class MQTTService:
    """
    MQTT Service managing consumer and publisher for:
      - Subscribed: <prefix>/data, <prefix>/actuator
      - Published:  <prefix>/result, <prefix>/cmd
    Uses Paho 2.x CallbackAPIVersion.VERSION2.
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.loop = loop
        self.prefix = settings.topic_prefix.rstrip("/")
        self.data_topic = f"{self.prefix}/data"
        self.actuator_topic = f"{self.prefix}/actuator"

        client_id = f"backend_consumer_{uuid.uuid4().hex[:8]}"
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.publisher = MQTTPublisher(self.client)
        self.ml_service = get_ml_service()
        self.repo = get_repository()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"[MQTT] Connected to broker ({settings.mqtt_broker}:{settings.mqtt_port}) with rc={rc}")
        topics = [
            (self.data_topic, 0),
            (self.actuator_topic, 0),
        ]
        client.subscribe(topics)
        logger.info(f"[MQTT] Subscribed to topics: {[t[0] for t in topics]}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        logger.warning(f"[MQTT] Disconnected from broker (rc={reason_code}). Auto-reconnect active...")

    def _on_message(self, client, userdata, message):
        """Dispatches incoming message to async coroutine in the event loop."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.handle_message(message.topic, message.payload),
                self.loop
            )
        else:
            asyncio.run(self.handle_message(message.topic, message.payload))

    async def handle_message(self, topic: str, payload_bytes: bytes):
        payload_str = payload_bytes.decode("utf-8", errors="replace")
        
        try:
            payload_dict = json.loads(payload_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[MQTT] Malformed JSON rejected on topic {topic}: {e} | Raw: {payload_str[:100]}")
            return

        if topic == self.data_topic:
            await self._handle_data_message(payload_dict)
        elif topic == self.actuator_topic:
            await self._handle_actuator_message(payload_dict, payload_str)
        else:
            logger.debug(f"[MQTT] Unhandled topic: {topic}")

    async def _handle_data_message(self, data: dict):
        ts = datetime.now(timezone.utc).isoformat()

        try:
            telemetry = TelemetryPayload.model_validate(data)
        except ValidationError as e:
            logger.warning(f"[MQTT] Telemetry validation error rejected: {e.errors()} | Input: {data}")
            return
        except Exception as e:
            logger.warning(f"[MQTT] Unexpected error validating telemetry payload: {e}")
            return

        device_id = telemetry.device_id or settings.device_id

        reading = ReadingRecord(
            ts=ts,
            device_id=device_id,
            hr=telemetry.hr,
            spo2=telemetry.spo2,
            temp=telemetry.temp,
            hum=telemetry.hum,
            room_temp=telemetry.room_temp,
            finger=telemetry.finger,
            fw_status=telemetry.status or "NORMAL"
        )
        try:
            await self.repo.save_reading(reading)
        except Exception as e:
            logger.error(f"[MQTT] Error in raw storage: {e}")

        if not telemetry.finger or not telemetry.has_valid_vitals:
            self.ml_service.reset_device(device_id)
            logger.info(f"[MQTT] Device {device_id}: finger={telemetry.finger}, vitals_valid={telemetry.has_valid_vitals}. /result omitted.")
            return

        rule_sev, rule_label = evaluate_rules(telemetry.hr, telemetry.spo2, telemetry.temp)

        ml_res = None
        try:
            ml_res = self.ml_service.process_reading(
                device_id=device_id,
                hr=telemetry.hr,
                spo2=telemetry.spo2,
                temp=telemetry.temp,
                finger=telemetry.finger
            )
        except Exception as e:
            logger.error(f"[MQTT] ML inference threw exception (continuing safely): {e}")

        arb = arbitrate(
            rule_severity=rule_sev,
            rule_label=rule_label,
            ml_result=ml_res,
            mode=settings.decision_mode
        )

        pred_record = PredictionRecord(
            ts=ts,
            device_id=device_id,
            rule_severity=arb.rule_severity,
            ml_severity=arb.ml_severity,
            ml_score=arb.ml_score,
            published_severity=arb.published_severity,
            decision_mode=arb.decision_mode,
            decision_source=arb.decision_source,
            label=arb.published_label,
            model_version=settings.model_version
        )
        try:
            await self.repo.save_prediction(pred_record)
        except Exception as e:
            logger.error(f"[MQTT] Error saving prediction record: {e}")

        result_payload = ResultPayload(
            label=arb.published_label,
            severity=arb.published_severity,
            score=arb.published_score,
            model_version=settings.model_version
        )
        self.publisher.publish_result(result_payload)

    async def _handle_actuator_message(self, data: dict, raw_str: str):
        ts = datetime.now(timezone.utc).isoformat()
        seq = data.get("ack_seq")
        try:
            seq_int = int(seq) if seq is not None else None
        except ValueError:
            seq_int = None

        event = ActuatorEventRecord(
            ts=ts,
            kind="status",
            seq=seq_int,
            payload_json=raw_str
        )
        try:
            await self.repo.save_actuator_event(event)
            logger.info(f"[MQTT] Recorded /actuator status (ack_seq={seq_int}, mode={data.get('mode')})")
        except Exception as e:
            logger.error(f"[MQTT] Error storing actuator status event: {e}")

    def start(self):
        """Connects and starts background MQTT loop."""
        try:
            self.client.connect(settings.mqtt_broker, settings.mqtt_port, keepalive=60)
            self.client.loop_start()
            logger.info(f"[MQTT] Background network loop started.")
        except Exception as e:
            logger.error(f"[MQTT] Connection failed to {settings.mqtt_broker}:{settings.mqtt_port}: {e}")

    def stop(self):
        """Stops loop and disconnects."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("[MQTT] Disconnected and loop stopped.")
        except Exception as e:
            logger.error(f"[MQTT] Error during disconnect: {e}")
