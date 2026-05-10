from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_version_shape() -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"app_version", "git_sha", "source_version", "build_ts"}
    assert data["app_version"] == "0.1.0"
