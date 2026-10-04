"""
src.api.routers.health
======================
GET  /health               Liveness check + known character list
DELETE /cache/{character}  Flush Redis cache (post-patch workflow)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import KNOWN_CHARACTERS, redis_client

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """
    Returns backend status, Redis connectivity, and the list of known characters.
    """
    redis_ok = False
    try:
        redis_client.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status":           "ok",
        "redis":            "connected" if redis_ok else "unavailable",
        "known_characters": sorted(set(KNOWN_CHARACTERS.values())),
    }


@router.delete("/cache/{character_name}", status_code=200)
def invalidate_cache(character_name: str) -> dict:
    """
    Flush the entire Redis cache.

    Called after re-ingesting a character post-patch.  Cache keys are SHA-256
    hashes of normalized questions, so surgical per-character deletion isn't
    possible without a secondary index.  For a low-traffic personal tool a
    full flush is acceptable — the cache will re-warm naturally.
    """
    char = character_name.lower()
    if char not in KNOWN_CHARACTERS:
        raise HTTPException(status_code=404, detail=f"Unknown character: {char!r}")

    redis_client.flushdb()
    return {"status": "cache cleared", "character": char}
