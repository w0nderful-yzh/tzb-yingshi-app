from app.infrastructure.external.ys7.alarm_mapper import Ys7AlarmMapper


def test_body_sensor_alarm_maps_to_person_detected() -> None:
    payload = Ys7AlarmMapper(default_device_serial="camera-01").map(
        {
            "alarmId": "alarm-person-01",
            "alarmType": 10000,
            "alarmStart": "2026-08-07 12:00:00",
            "alarmPicUrl": "https://example.invalid/person.jpg",
        }
    )

    assert payload is not None
    assert payload["eventType"] == "person_detected"
    assert payload["deviceId"] == "camera-01"
    assert payload["timestamp"] == "2026-08-07T04:00:00+00:00"
    assert payload["imageUrl"] == "https://example.invalid/person.jpg"


def test_explicit_phone_call_text_maps_to_phone_call() -> None:
    payload = Ys7AlarmMapper(default_device_serial="camera-01").map(
        {
            "alarmId": "alarm-phone-01",
            "alarmType": "custom-ai",
            "alarmName": "打电话检测",
            "alarmStart": 1_786_069_200_000,
            "confidence": 93,
        }
    )

    assert payload is not None
    assert payload["eventType"] == "phone_call"
    assert payload["confidence"] == 0.93


def test_people_count_has_its_own_event_type() -> None:
    payload = Ys7AlarmMapper(default_device_serial="camera-01").map(
        {
            "alarmId": "alarm-count-01",
            "alarmType": "scene-statistics",
            "peopleCount": 2,
        }
    )

    assert payload is not None
    assert payload["eventType"] == "people_count"
    assert payload["peopleCount"] == 2


def test_generic_motion_alarm_is_not_claimed_as_person_detection() -> None:
    payload = Ys7AlarmMapper(default_device_serial="camera-01").map(
        {
            "alarmId": "alarm-motion-01",
            "alarmType": 10002,
            "alarmName": "移动侦测",
        }
    )

    assert payload is None
