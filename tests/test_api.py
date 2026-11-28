import json

import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

VALID_PAYLOAD = {
    "students_on_campus": 1200,
    "menu_price_rs": 80,
    "temp_c": 31.5,
    "exam_days_left": 12,
    "special_dishes": 2,
    "queue_minutes_yesterday": 18,
    "day_type": "weekday",
    "block": "A-Block",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_uses_model_and_returns_prediction(monkeypatch):
    class StubModel:
        def predict(self, frame):
            assert list(frame.columns) == main.FEATURES
            assert frame.loc[0, "students_on_campus"] == 1200
            return [742.5]

    monkeypatch.setattr(main, "_model", StubModel())
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"prediction": 742.5}


@pytest.mark.parametrize(
    "changes",
    [
        {"menu_price_rs": 201},
        {"students_on_campus": 399},
        {"unexpected": "field"},
        {"day_type": "sunday"},
    ],
)
def test_predict_rejects_invalid_request_with_422(changes):
    payload = {**VALID_PAYLOAD, **changes}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_logs_json(monkeypatch, caplog):
    class StubModel:
        def predict(self, frame):
            return [700]

    monkeypatch.setattr(main, "_model", StubModel())
    with caplog.at_level("INFO", logger="mp033.api"):
        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    event = json.loads(caplog.records[-1].message)
    assert event["event"] == "prediction"
    assert event["prediction"] == 700.0
