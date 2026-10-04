"""
src.api.dependencies
====================
Module-level singletons shared across routers.

Clients are initialised once at import time so they're reused across
requests (connection pooling). The character registry is rebuilt from
``character_data/`` at startup, giving instant known-character lookups
without a database round-trip.

Character registry design
--------------------------
Keys are every recognisable name variant a user might type; values are
the exact ``character`` metadata string stored in Pinecone (the full
lowercased ``character_name`` from the JSON).

Example entries for Tartaglia::

    "tartaglia (childe)"  → "tartaglia (childe)"   # full name
    "tartaglia"           → "tartaglia (childe)"   # first word
    "childe"              → "tartaglia (childe)"   # last word (alias)

Lookup always returns the Pinecone metadata value, which is what the
filter passed to ``index.search()`` must equal.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import redis as redis_lib
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.environ.get("PINECONE_INDEX_NAME", "genshin-assistant")
REDIS_URL        = os.environ.get("REDIS_URL",           "redis://localhost:6379")
REDIS_TTL        = int(os.environ.get("REDIS_TTL",       86400))

CHARACTER_DATA_DIR = Path("character_data")

GENERATION_MODEL = "gemini-2.0-flash"
RETRIEVAL_TOP_K  = 5

#: Namespace used in both the ingest pipeline and retrieval queries.
NAMESPACE = "characters"

# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------

gemini_client: genai.Client   = genai.Client(api_key=GEMINI_API_KEY)
pinecone_idx                  = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
redis_client: redis_lib.Redis = redis_lib.from_url(REDIS_URL, decode_responses=True)

# ---------------------------------------------------------------------------
# Character registry
# ---------------------------------------------------------------------------

# Single-word primaries too generic to trigger a character lookup
_SKIP_PRIMARIES: frozenset[str] = frozenset({
    "hu",     # needs "hu tao" (two words)
    "lan",    # needs "lan yan"
    "yun",    # needs "yun jin"
    "cryo",   # element name — would match non-character questions
    "hydro",
    "dendro",
    "pyro",
})


def _build_character_registry() -> dict[str, str]:
    """
    Scan ``character_data/`` and build a lookup dict::

        variant_name → pinecone_metadata_value

    Each character contributes up to three entries:
    - full lowercased character_name (always)
    - first word  (unless in _SKIP_PRIMARIES or already registered)
    - last word stripped of parens (e.g. "childe", "heizou", "miko")
    """
    registry: dict[str, str] = {}

    for f in CHARACTER_DATA_DIR.glob("*.json"):
        try:
            data   = json.loads(f.read_text(encoding="utf-8"))
            full   = (data.get("character_name") or f.stem).lower()

            # Skip legacy files whose name is a page title not a character name
            if "rating and information" in full or "best builds" in full:
                primary = full.split()[0]
                registry.setdefault(primary, primary)
                continue

            registry[full] = full

            # First word
            first = re.split(r"[\s(]", full)[0].strip()
            if first and first != full and first not in _SKIP_PRIMARIES:
                registry.setdefault(first, full)

            # Last word (strip surrounding parens)
            words     = full.split()
            last      = words[-1].strip("()") if words else ""
            if (last and last != first and last != full
                    and len(last) >= 4
                    and last not in _SKIP_PRIMARIES):
                registry.setdefault(last, full)

        except Exception:
            pass

    return registry


#: Built once at startup; used by routers for character name resolution.
KNOWN_CHARACTERS: dict[str, str] = _build_character_registry()


def resolve_character(question: str) -> str | None:
    """
    Return the Pinecone ``character`` metadata value for the character named
    in *question*, or ``None`` if no known character is mentioned.

    Iterates longest key first so multi-word names ("hu tao", "yae miko")
    match before single-word substrings ("hu").
    """
    q = question.lower()
    for variant in sorted(KNOWN_CHARACTERS, key=len, reverse=True):
        if variant in q:
            return KNOWN_CHARACTERS[variant]
    return None
