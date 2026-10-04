"""
src.api.routers.query
=====================
POST /query — the main RAG pipeline endpoint.

Pipeline
--------
1. Normalize question → check Redis cache → return on hit
2. Resolve character name → known-character fallback guard
3. Search Pinecone (server-side embedding, top-k=5)
4. Guard against empty retrieval
5. Generate answer via Gemini with a grounding system instruction
6. Write answer to Redis cache
7. Return answer + source metadata + latency
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from fastapi import APIRouter, HTTPException
from google.genai import types as gtypes
from pinecone.db_data.dataclasses.search_query import SearchQuery
from pydantic import BaseModel

from ..dependencies import (
    GENERATION_MODEL,
    KNOWN_CHARACTERS,
    NAMESPACE,
    REDIS_TTL,
    RETRIEVAL_TOP_K,
    gemini_client,
    pinecone_idx,
    redis_client,
    resolve_character,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    character:    str
    section:      str
    last_updated: str
    score:        float


class QueryResponse(BaseModel):
    answer:     str
    sources:    list[Source]
    cached:     bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Prompt / system instruction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a Genshin Impact build assistant. Answer questions about character builds,
artifacts, weapons, team compositions, talent priority, and materials.

Rules:
- Answer ONLY from the retrieved context provided in the user message.
- Do not use outside knowledge or invent information absent from the context.
- If the context does not contain enough information, say so explicitly — do not guess.
- Always cite which character and section your answer came from
  (e.g. "According to Varka's artifacts data, ...").
- Be concise and direct. Use bullet points for lists.
- Quote stat values, item names, and numbers exactly as given in the context.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Words that look like proper nouns but are not character names — used in the
# unknown-character fallback heuristic.
_COMMON_WORDS: frozenset[str] = frozenset({
    "what", "who",   "how",   "when",  "where", "why",   "which",
    "the",  "a",     "an",    "is",    "are",   "do",    "does",
    "can",  "best",  "good",  "for",   "with",  "his",   "her",
    "their","my",    "your",  "team",  "build", "guide", "list",
    "show", "give",  "tell",  "help",  "use",   "need",  "want",
})

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?)\b")


def _normalize(q: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for cache keying."""
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    return re.sub(r"\s+", " ", q)


def _cache_key(normalized: str) -> str:
    return "genshin::" + hashlib.sha256(normalized.encode()).hexdigest()


def _is_unknown_character_question(question: str, resolved: str | None) -> bool:
    """
    Return True if the question names a proper noun that isn't in
    KNOWN_CHARACTERS, suggesting the user is asking about a character we
    don't have data for.
    """
    if resolved:
        return False  # already resolved — not unknown
    words = [w.lower() for w in _PROPER_NOUN_RE.findall(question)]
    unknown = [w for w in words if w not in _COMMON_WORDS and w not in KNOWN_CHARACTERS]
    return bool(unknown)


def _fallback_message(question: str) -> str:
    for word in _PROPER_NOUN_RE.findall(question):
        if word.lower() not in KNOWN_CHARACTERS:
            return (
                f"I don't have build data for **{word}** yet. "
                "They may not have been scraped or may be an upcoming character. "
                "Try asking about a character I know, like Varka or Neuvillette."
            )
    return (
        "I couldn't identify which character you're asking about, "
        "or I don't have data for them yet. Please name the character in your question."
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve(question: str, character_filter: str | None) -> tuple[list[str], list[Source]]:
    query = SearchQuery(
        inputs={"text": question},
        top_k=RETRIEVAL_TOP_K,
        filter={"character": {"$eq": character_filter}} if character_filter else None,
    )
    results = pinecone_idx.search(
        namespace=NAMESPACE,
        query=query,
        fields=["text", "character", "section", "last_updated"],
    )
    hits = results.to_dict().get("result", {}).get("hits", [])

    chunks:  list[str]    = []
    sources: list[Source] = []
    for hit in hits:
        fields = hit.get("fields", {})
        text   = fields.get("text", "")
        if text:
            chunks.append(text)
        sources.append(Source(
            character    = fields.get("character",    ""),
            section      = fields.get("section",      ""),
            last_updated = fields.get("last_updated", ""),
            score        = round(hit.get("_score", 0.0), 4),
        ))

    return chunks, sources


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt  = f"Retrieved context:\n\n{context}\n\n---\n\nQuestion: {question}"
    resp    = gemini_client.models.generate_content(
        model    = GENERATION_MODEL,
        contents = prompt,
        config   = gtypes.GenerateContentConfig(
            system_instruction = _SYSTEM_INSTRUCTION,
            temperature        = 0.2,
            max_output_tokens  = 1024,
        ),
    )
    return resp.text


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    normalized = _normalize(question)

    # 1. Cache check
    key       = _cache_key(normalized)
    cached_raw = redis_client.get(key)
    if cached_raw:
        payload = json.loads(cached_raw)
        return QueryResponse(
            answer     = payload["answer"],
            sources    = [Source(**s) for s in payload["sources"]],
            cached     = True,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 2. Character resolution + fallback guard
    char_filter = resolve_character(question)
    if _is_unknown_character_question(question, char_filter):
        return QueryResponse(
            answer     = _fallback_message(question),
            sources    = [],
            cached     = False,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 3. Retrieve
    chunks, sources = _retrieve(question, char_filter)

    if not chunks:
        known = ", ".join(sorted(set(KNOWN_CHARACTERS.values())))
        return QueryResponse(
            answer = (
                "I couldn't find relevant build data for your question. "
                "Make sure the character has been scraped and ingested.\n\n"
                f"Known characters: {known}."
            ),
            sources    = [],
            cached     = False,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 4. Generate
    answer = _generate(question, chunks)

    # 5. Cache
    redis_client.setex(key, REDIS_TTL, json.dumps({
        "answer":  answer,
        "sources": [s.model_dump() for s in sources],
    }))

    return QueryResponse(
        answer     = answer,
        sources    = sources,
        cached     = False,
        latency_ms = round((time.monotonic() - t0) * 1000, 1),
    )
