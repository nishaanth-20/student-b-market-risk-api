"""Offline tests: Kraken and the peer APIs are replaced with fake responses.

Run from the repo root:  python -m pytest -q
"""

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.data as data
import app.main as main
import app.peers as peers


def fake_kraken_payload(days: int = 60) -> dict:
    """Kraken-shaped OHLC: completed days plus today's unfinished candle."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for i in range(days, -1, -1):  # i = 0 is today (unfinished)
        day = today - timedelta(days=i)
        price = 100 + (days - i)  # rises by 1 per day
        rows.append([int(day.timestamp()), str(price), str(price + 2), str(price - 2),
                     str(price + 0.5), str(price), "1000", 10])
    return {"error": [], "result": {"XXBTZUSD": rows, "last": rows[-1][0]}}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def fake_kraken(monkeypatch):
    data._cache.clear()
    monkeypatch.setattr(data.httpx, "get", lambda *a, **k: FakeResponse(fake_kraken_payload()))


@pytest.fixture
def client():
    return TestClient(main.app)


def last_completed_date() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


def test_root_and_health(client):
    assert client.get("/").json()["endpoints"] == ["/", "/health", "/features", "/predict", "/model-metadata"]
    assert client.get("/health").json()["status"] == "ok"


def test_features_format(client):
    body = client.post("/features", json={"token": "btc"}).json()
    assert list(body)[:2] == ["token", "date"]
    assert body["token"] == "BTC"
    assert body["date"] == last_completed_date()  # unfinished candle dropped
    # closes are 100.5, 101.5, ... so the last 5 closes average to last - 2
    last_close = 100.5 + 59
    assert body["MA5"] == pytest.approx(last_close - 2)
    assert body["MA20"] == pytest.approx(last_close - 9.5)
    assert body["TrendStrength"] == pytest.approx((last_close - 2) / (last_close - 9.5) - 1)


def test_bad_token_and_horizon(client):
    r = client.post("/features", json={"token": "ABC"})
    assert r.status_code == 422 and r.json()["supported_tokens"][0] == "BTC"
    r = client.post("/predict", json={"token": "BTC", "forecast_horizon": 5})
    assert r.status_code == 422 and "error" in r.json()
    r = client.post("/predict", json={})  # missing token -> same format, not FastAPI's default
    assert r.status_code == 422 and "supported_tokens" in r.json()


def test_kraken_down_gives_503(client, monkeypatch):
    def boom(*a, **k):
        raise data.httpx.ConnectError("down")
    monkeypatch.setattr(data.httpx, "get", boom)
    r = client.post("/features", json={"token": "ETH"})
    assert r.status_code == 503 and "error" in r.json()


class ConstantModel:
    def predict(self, X):
        self.last_X = X
        return np.array([42.0])


def test_predict_with_peers_down(client, monkeypatch):
    monkeypatch.setattr(main, "model", ConstantModel())

    async def all_down(token):
        return {"student_a": {"ok": False, "error": "timed out after 10s"},
                "student_c": {"ok": False, "error": "URL not configured"}}
    monkeypatch.setattr(main, "fetch_peer_features", all_down)

    r = client.post("/predict", json={"token": "BTC", "forecast_horizon": 7})
    body = r.json()
    assert r.status_code == 200
    assert body["prediction"] == 42.0
    assert len(body["missing_features"]) == 5
    assert body["prediction_date"] == (datetime.now(timezone.utc).date() + timedelta(days=6)).isoformat()


def test_predict_with_peers_up(client, monkeypatch):
    m = ConstantModel()
    monkeypatch.setattr(main, "model", m)
    date = last_completed_date()

    async def all_up(token):
        return {"student_a": {"ok": True, "data": {"token": token, "date": date, "Return20": -0.15}},
                "student_c": {"ok": True, "data": {"token": token, "date": date, "Volatility20": 0.05,
                                                    "MaxPrice20": 200.0, "GapRisk": 0.01}}}
    monkeypatch.setattr(main, "fetch_peer_features", all_up)

    body = client.post("/predict", json={"token": "BTC"}).json()
    assert body["missing_features"] == []
    close = 100.5 + 59
    assert body["features_used"]["DrawdownRisk"] == pytest.approx((200 - close) / 200)
    assert list(m.last_X.columns) == main.MODEL_FEATURES
    assert body["current_mri"] is not None


def test_peer_with_wrong_date_is_ignored(client, monkeypatch):
    monkeypatch.setattr(main, "model", ConstantModel())

    async def stale(token):
        return {"student_a": {"ok": True, "data": {"date": "2020-01-01", "Return20": 0.1}},
                "student_c": {"ok": False, "error": "x"}}
    monkeypatch.setattr(main, "fetch_peer_features", stale)

    body = client.post("/predict", json={"token": "BTC"}).json()
    assert "Return20" in body["missing_features"] and body["warnings"]


def test_peer_calls_run_in_parallel(monkeypatch):
    """Two slow peers should take ~1 delay, not 2."""
    import asyncio

    async def slow(client, name, token):
        await asyncio.sleep(0.5)
        return name, {"ok": True, "data": {}}
    monkeypatch.setattr(peers, "_fetch_one", slow)
    start = time.time()
    asyncio.run(peers.fetch_peer_features("BTC"))
    assert time.time() - start < 0.9
