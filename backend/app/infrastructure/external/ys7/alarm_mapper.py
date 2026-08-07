import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

PERSON_ALARM_TYPE_CODES = {"10000", "10010", "10015"}
PERSON_TERMS = (
    "person",
    "human",
    "body_sensor",
    "face_detection",
    "人体",
    "人形",
    "人员",
    "人脸",
)
PHONE_CALL_TERMS = (
    "phone_call",
    "phonecall",
    "telephone",
    "calling",
    "打电话",
    "通话",
    "电话检测",
)


class Ys7AlarmMapper:
    def __init__(self, *, default_device_serial: str) -> None:
        self._default_device_serial = default_device_serial

    def map(self, alarm: dict[str, object]) -> dict[str, object] | None:
        alarm_id = self._first_text(alarm, "alarmId", "alarm_id", "id")
        if alarm_id is None:
            return None
        event_type = self._event_type(alarm)
        if event_type is None:
            return None
        device_serial = (
            self._first_text(alarm, "deviceSerial", "deviceId", "device_id")
            or self._default_device_serial
        )
        payload: dict[str, object] = {
            "messageId": alarm_id,
            "eventId": alarm_id,
            "deviceId": device_serial,
            "timestamp": self._event_time(alarm).isoformat(),
            "eventType": event_type,
            "vendorAlarm": alarm,
        }
        image_url = self._first_text(
            alarm,
            "alarmPicUrl",
            "picUrl",
            "imageUrl",
            "pictureUrl",
        )
        if image_url is not None:
            payload["imageUrl"] = image_url
        confidence = self._confidence(alarm)
        if confidence is not None:
            payload["confidence"] = confidence
        people_count = self._people_count(alarm)
        if people_count is not None:
            payload["peopleCount"] = people_count
        return payload

    def alarm_type_label(self, alarm: dict[str, object]) -> str:
        return (
            self._first_text(
                alarm,
                "alarmType",
                "alarm_type",
                "eventType",
                "event_type",
                "alarmName",
            )
            or "unknown"
        )

    def _event_type(self, alarm: dict[str, object]) -> str | None:
        searchable = " ".join(self._searchable_values(alarm)).lower()
        if any(term in searchable for term in PHONE_CALL_TERMS):
            return "phone_call"
        if self._people_count(alarm) is not None:
            return "people_count"
        alarm_type = self._first_text(
            alarm,
            "alarmType",
            "alarm_type",
            "eventType",
            "event_type",
        )
        if alarm_type in PERSON_ALARM_TYPE_CODES:
            return "person_detected"
        if any(term in searchable for term in PERSON_TERMS):
            return "person_detected"
        return None

    def _event_time(self, alarm: dict[str, object]) -> datetime:
        value = self._first_value(
            alarm,
            "alarmStart",
            "alarmTime",
            "alarm_time",
            "createTime",
            "create_time",
            "startTime",
            "time",
        )
        if isinstance(value, (int, float)):
            seconds = float(value) / 1000 if value > 10_000_000_000 else float(value)
            return datetime.fromtimestamp(seconds, tz=UTC)
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                numeric = int(text)
                seconds = numeric / 1000 if numeric > 10_000_000_000 else numeric
                return datetime.fromtimestamp(seconds, tz=UTC)
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
                return parsed.astimezone(UTC)
            except ValueError:
                pass
        return datetime.now(UTC)

    def _people_count(self, alarm: dict[str, object]) -> int | None:
        value = self._first_value(
            alarm,
            "peopleCount",
            "people_count",
            "personCount",
            "person_count",
        )
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed is not None and parsed >= 0 else None

    def _confidence(self, alarm: dict[str, object]) -> float | None:
        value = self._first_value(alarm, "confidence", "score", "probability")
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        if parsed > 1 and parsed <= 100:
            parsed /= 100
        return parsed if parsed <= 1 else None

    def _searchable_values(self, alarm: dict[str, object]) -> list[str]:
        values: list[str] = []
        for key in (
            "alarmType",
            "alarm_type",
            "eventType",
            "event_type",
            "alarmName",
            "alarmContent",
            "content",
            "customerType",
            "customerInfo",
        ):
            value = alarm.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
        return values

    @staticmethod
    def _first_value(alarm: dict[str, object], *keys: str) -> object | None:
        for key in keys:
            value = alarm.get(key)
            if value is not None:
                return value
        return None

    @classmethod
    def _first_text(cls, alarm: dict[str, object], *keys: str) -> str | None:
        value = cls._first_value(alarm, *keys)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
