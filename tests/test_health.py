"""
tests/test_health.py
====================
Tests for the ops endpoints: GET /health and DELETE /cache/{name}.
"""

from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "varka" in body["known_characters"]
    assert body["redis"] in ("connected", "unavailable")


def test_cache_flush_known_character(client, deps):
    deps.redis_client.flushdb.return_value = True
    resp = client.delete("/cache/varka")
    assert resp.status_code == 200
    assert resp.json()["character"] == "varka"
    deps.redis_client.flushdb.assert_called_once()


def test_cache_flush_unknown_character_404(client):
    resp = client.delete("/cache/nonexistard")
    assert resp.status_code == 404
