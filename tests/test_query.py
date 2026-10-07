"""
tests/test_query.py
===================
Behavioural tests for POST /query, mirroring the four cases in CLAUDE.md §7:

1. Known character + valid question → grounded answer with sources
2. Unknown / unreleased character → explicit fallback, LLM never called
3. Cache hit → cached answer returned, LLM never called
4. Empty / malformed question → 400

All external calls (Gemini, Pinecone, Redis) are mocked; the suite runs offline.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.api.routers.query import Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_sources():
    return [
        Source(character="varka", section="artifacts", last_updated="2026-08-26", score=0.91),
    ]


def _fake_chunks():
    return ["Artifact builds for Varka: [Best-in-Slot] 4-pc A Day Carved from Rising Winds."]


# ---------------------------------------------------------------------------
# 1. Known character + valid question → grounded answer + sources
# ---------------------------------------------------------------------------

def test_known_character_returns_answer_and_sources(client, query_mod, monkeypatch):
    monkeypatch.setattr(query_mod, "_semantic_cache_lookup", lambda q: None)
    monkeypatch.setattr(query_mod, "_retrieve",
                        lambda q, c, s: (_fake_chunks(), _fake_sources()))
    gen = MagicMock(return_value="Varka's artifacts data shows A Day Carved from Rising Winds.")
    monkeypatch.setattr(query_mod, "_generate", gen)

    resp = client.post("/query", json={"question": "best artifacts for Varka?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "A Day Carved" in body["answer"]
    assert body["cached"] is False
    assert len(body["sources"]) == 1
    assert body["sources"][0]["character"] == "varka"
    assert body["sources"][0]["section"] == "artifacts"
    gen.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Unknown character → fallback message, LLM NOT called
# ---------------------------------------------------------------------------

def test_unknown_character_triggers_fallback_without_llm(client, query_mod, monkeypatch):
    monkeypatch.setattr(query_mod, "_semantic_cache_lookup", lambda q: None)

    retrieve = MagicMock()
    generate = MagicMock()
    monkeypatch.setattr(query_mod, "_retrieve", retrieve)
    monkeypatch.setattr(query_mod, "_generate", generate)

    # "Nonexistard" is a capitalised proper noun not in the registry
    resp = client.post("/query", json={"question": "best build for Nonexistard?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "don't have build data" in body["answer"].lower()
    assert body["sources"] == []
    # Critical: neither retrieval nor generation ran
    retrieve.assert_not_called()
    generate.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Cache hit → cached answer, LLM NOT called
# ---------------------------------------------------------------------------

def test_cache_hit_returns_cached_answer_without_llm(client, deps, query_mod, monkeypatch):
    cached_payload = {
        "answer": "Cached: Varka wants A Day Carved from Rising Winds.",
        "sources": [s.model_dump() for s in _fake_sources()],
    }
    deps.redis_client.get.return_value = json.dumps(cached_payload)

    generate = MagicMock()
    retrieve = MagicMock()
    monkeypatch.setattr(query_mod, "_generate", generate)
    monkeypatch.setattr(query_mod, "_retrieve", retrieve)

    resp = client.post("/query", json={"question": "best artifacts for Varka?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert body["answer"].startswith("Cached:")
    generate.assert_not_called()
    retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Empty / malformed question → 400
# ---------------------------------------------------------------------------

def test_empty_question_returns_400(client):
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_missing_question_field_returns_422(client):
    # Pydantic rejects a body without the required field before our handler runs
    resp = client.post("/query", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 5. Empty retrieval → descriptive message, no hallucination, LLM NOT called
# ---------------------------------------------------------------------------

def test_empty_retrieval_returns_no_data_message(client, query_mod, monkeypatch):
    monkeypatch.setattr(query_mod, "_semantic_cache_lookup", lambda q: None)
    monkeypatch.setattr(query_mod, "_retrieve", lambda q, c, s: ([], []))
    generate = MagicMock()
    monkeypatch.setattr(query_mod, "_generate", generate)

    resp = client.post("/query", json={"question": "tell me about Varka"})

    assert resp.status_code == 200
    body = resp.json()
    assert "couldn't find relevant build data" in body["answer"].lower()
    generate.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Generation error → graceful message, not a 500
# ---------------------------------------------------------------------------

def test_generation_error_returns_graceful_message(client, query_mod, monkeypatch):
    monkeypatch.setattr(query_mod, "_semantic_cache_lookup", lambda q: None)
    monkeypatch.setattr(query_mod, "_retrieve",
                        lambda q, c, s: (_fake_chunks(), _fake_sources()))

    def _boom(*a, **k):
        raise RuntimeError("503 UNAVAILABLE")
    monkeypatch.setattr(query_mod, "_generate", _boom)

    resp = client.post("/query", json={"question": "best artifacts for Varka?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body["answer"].lower()
