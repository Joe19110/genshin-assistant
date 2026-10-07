"""
tests/conftest.py
=================
Shared pytest fixtures.

The API's ``dependencies`` module instantiates the Gemini, Pinecone, and
Redis clients at *import time* and reads API keys from the environment.
To let the suite run completely offline we:

1. Set dummy env vars before anything imports the app.
2. Monkeypatch the client constructors so no real network clients are built.
3. Expose fixtures that patch the module-level singletons / helpers per test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the project root importable (tests/ is one level down)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# IMPORTANT: stub env + client constructors at *import time*, before any test
# module (or the app) imports src.api.dependencies. Doing this inside a fixture
# is too late — dependencies.py builds its client singletons on import.
# ---------------------------------------------------------------------------

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("GEMINI_MODEL", "gemini-test")

import google.genai as _genai
import redis as _redis_lib
import pinecone as _pinecone

_genai.Client = MagicMock(return_value=MagicMock(name="gemini_client"))
_redis_lib.from_url = MagicMock(return_value=MagicMock(name="redis_client"))

_mock_pc = MagicMock(name="pinecone")
_mock_pc.Index.return_value = MagicMock(name="pinecone_index")
_pinecone.Pinecone = MagicMock(return_value=_mock_pc)


# ---------------------------------------------------------------------------
# Per-test client with patchable dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def deps():
    """Return the imported dependencies module (clients are already mocks)."""
    from src.api import dependencies
    return dependencies


@pytest.fixture
def query_mod():
    """Return the query router module so tests can patch its helpers."""
    from src.api.routers import query
    return query


@pytest.fixture
def client(deps):
    """
    FastAPI TestClient with:
    - a known-character registry seeded with Varka
    - Redis cache mocked to always-miss by default
    """
    from fastapi.testclient import TestClient
    from src.api.app import app

    # Seed a predictable character registry
    deps.KNOWN_CHARACTERS.clear()
    deps.KNOWN_CHARACTERS.update({"varka": "varka"})

    # Default Redis behaviour: cache miss, writes accepted
    deps.redis_client.get.return_value = None
    deps.redis_client.setex.return_value = True
    deps.redis_client.ping.return_value = True

    return TestClient(app)
