import json
import logging
from datetime import datetime, timezone
from typing import Optional
import paho.mqtt.client as mqtt
from app.config import settings
from app.schemas.result import ResultPayload
from app.schemas.cmd import CommandPayload
from app.mqtt.seq_manager import get_seq_manager
from app.database.interfaces import ActuatorEventRecord
from app.database.factory import get_repository

logger = logging.getLogger(__name__)

class MQTTPublisher:
    """
    Handles publishing to <prefix>/result and <prefix>/cmd.
    """

    def __init__(self, client: mqtt.Client):
        self.client = client
        self.prefix = settings.topic_prefix.rstrip("/")
        self.result_topic = f"{self.prefix}/result"
        self.cmd_topic = f"{self.prefix}/cmd"
        self.seq_manager = get_seq_manager()

    def publish_result(self, result: ResultPayload) -> bool:
        """
        Publishes ML / Rule result payload to <prefix>/result.
        Format: {"label":"NORMAL","severity":0,"score":0.12,"model_version":"if_v1"}
        """
        try:
            payload_str = result.model_dump_json()
            info = self.client.publish(self.result_topic, payload_str, qos=0)
            logger.info(f"[MQTTPublisher] Published /result: {payload_str}")
            return info.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"[MQTTPublisher] Error publishing /result: {e}")
            return False

    async def publish_command(
        self,
        mode: Optional[str] = None,
        color: Optional[str] = None,
        brightness: Optional[int] = None,
        buzzer_test: Optional[bool] = None
    ) -> Optional[CommandPayload]:
        """
        Generates next 32-bit seq, builds CommandPayload, publishes to <prefix>/cmd,
        and saves event to actuator_events.csv.
        """
        seq = self.seq_manager.next_seq()
        cmd = CommandPayload(
            mode=mode,
            color=color,
            brightness=brightness,
            buzzer_test=buzzer_test,
            seq=seq
        )
        payload_str = cmd.model_dump_json(exclude_none=True)

        try:
            self.client.publish(self.cmd_topic, payload_str, qos=0)
            logger.info(f"[MQTTPublisher] Published /cmd (seq={seq}): {payload_str}")

            # Persist command event
            ts = datetime.now(timezone.utc).isoformat()
            repo = get_repository()
            await repo.save_actuator_event(ActuatorEventRecord(
                ts=ts,
                kind="cmd",
                seq=seq,
                payload_json=payload_str
            ))
            return cmd
        except Exception as e:
            logger.error(f"[MQTTPublisher] Error publishing /cmd: {e}")
            return None
