"""
src.api.routers.query
=====================
POST /query — the main RAG pipeline endpoint.

Pipeline
--------
1. Normalize question → check Redis cache → return on hit
2. Resolve character name → known-character fallback guard
3. Detect section intent → optionally filter Pinecone by section
4. Search Pinecone (server-side embedding, top-k=5)
5. Guard against empty retrieval
6. Build a labelled context block → generate via Gemini
7. Write answer + sources to Redis cache
8. Return answer + source metadata + latency
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
    CACHE_NAMESPACE,
    CACHE_SIMILARITY_THRESHOLD,
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
# System instruction
# ---------------------------------------------------------------------------
# Kept deliberately short — every token here is billed on every request.
# Domain knowledge (stat rules, reaction mechanics) lives in the retrieved
# chunks, not here.  This instruction only sets behaviour, not content.

_SYSTEM_INSTRUCTION = """\
You are a Genshin Impact build assistant. Answer questions using ONLY the
retrieved context provided. Do not use outside knowledge or invent details.

Rules:
- If the context doesn't cover the question, say so — do not guess.
- Cite the character and section (e.g. "Varka's artifacts data shows...").
- Use bullet points for stat lists, team members, and ranked items.
- Quote numbers and item names exactly as they appear in the context.
- Be concise. Skip filler phrasing and unnecessary restatements of the question.
"""

# ---------------------------------------------------------------------------
# Section intent detection
# ---------------------------------------------------------------------------

# Maps question keywords → Pinecone section name.
# Used to add an optional section filter so the most relevant chunk
# ranks first even when retrieval would otherwise pull other sections.
_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(artifact|set bonus|sands|goblet|circlet|substat|main stat"
                r"|crit|atk%|hp%|def%|elemental mastery|energy recharge"
                r"|stat priority|goal stat)\b", re.I), "artifacts"),
    (re.compile(r"\b(weapon|sword|claymore|polearm|bow|catalyst|passive|r1|r5"
                r"|refinement|best.?in.?slot|bis|f2p weapon|4.?star weapon)\b", re.I), "weapons"),
    (re.compile(r"\b(team|comp|partner|pair|synerg|who (should|do) i|alongside"
                r"|mono.?(pyro|hydro|electro|cryo|anemo|geo|dendro)"
                r"|support for|run with)\b", re.I), "teams"),
    (re.compile(r"\b(talent|skill|burst|normal attack|level up|priority"
                r"|passive|ascension passive|kit|ability|mechanic)\b", re.I), "talents"),
    (re.compile(r"\b(constellation|c[1-6]\b|c0|pull|worth it|skip|crown)\b", re.I), "constellations"),
    (re.compile(r"\b(material|ascend|ascension|talent book|book|boss drop"
                r"|mora|local specialty|farm|where (do|can) i get)\b", re.I), "materials"),
    (re.compile(r"\b(good|worth|tier|rating|role|strong|weak|meta|viable"
                r"|recommend|overview|summary|how is)\b", re.I), "overview"),
]


def _detect_section(question: str) -> str | None:
    """
    Return the most likely section the question is asking about, or None
    if the question spans multiple sections or is ambiguous.
    """
    matches: dict[str, int] = {}
    for pattern, section in _SECTION_PATTERNS:
        count = len(pattern.findall(question))
        if count:
            matches[section] = matches.get(section, 0) + count

    if not matches:
        return None
    top_section, top_count = max(matches.items(), key=lambda x: x[1])
    # Only apply a section filter if one section clearly dominates
    total = sum(matches.values())
    return top_section if top_count / total >= 0.6 else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_WORDS: frozenset[str] = frozenset({
    "what", "who",   "how",   "when",  "where", "why",   "which",
    "the",  "a",     "an",    "is",    "are",   "do",    "does",
    "can",  "best",  "good",  "for",   "with",  "his",   "her",
    "their","my",    "your",  "team",  "build", "guide", "list",
    "show", "give",  "tell",  "help",  "use",   "need",  "want",
})

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?)\b")


def _normalize(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    return re.sub(r"\s+", " ", q)


def _cache_key(normalized: str) -> str:
    return "genshin::" + hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Semantic cache (two-layer: exact Redis hit → Pinecone similarity hit → generate)
# ---------------------------------------------------------------------------

def _semantic_cache_lookup(question: str) -> tuple[str, list[dict]] | None:
    """
    Search the ``answer_cache`` Pinecone namespace for a past question
    whose embedding is similar enough to *question*.  If found, return the
    cached (answer, sources) tuple; otherwise return None.

    This catches rephrased questions ("best artifacts for Varka" vs
    "what artifacts should I use on Varka") that the exact Redis hash
    would miss, saving a Gemini API call.
    """
    try:
        results = pinecone_idx.search(
            namespace=CACHE_NAMESPACE,
            query=SearchQuery(
                inputs={"text": question},
                top_k=1,
            ),
            fields=["cached_answer", "cached_sources"],
        )
        hits = results.to_dict().get("result", {}).get("hits", [])
        if not hits:
            return None

        hit = hits[0]
        score = hit.get("_score", 0.0)
        if score < CACHE_SIMILARITY_THRESHOLD:
            return None

        fields = hit.get("fields", {})
        answer  = fields.get("cached_answer")
        sources = fields.get("cached_sources")

        if answer and sources:
            # sources is stored as a JSON string by Pinecone metadata
            if isinstance(sources, str):
                sources = json.loads(sources)
            return answer, sources

    except Exception:
        pass  # cache failures should never block the main pipeline

    return None


def _semantic_cache_store(question: str, answer: str, sources: list[Source]) -> None:
    """
    Store a generated answer in the ``answer_cache`` Pinecone namespace so
    that future semantically similar questions can hit the cache instead of
    calling Gemini again.

    Uses the same Pinecone index — Pinecone server-side embedding handles
    the question text → vector conversion automatically.
    """
    try:
        record = {
            "_id":           _cache_key(_normalize(question)),
            "text":          question,
            "cached_answer": answer,
            "cached_sources": json.dumps([s.model_dump() for s in sources]),
        }
        pinecone_idx.upsert_records(namespace=CACHE_NAMESPACE, records=[record])
    except Exception:
        pass  # best-effort — don't block the response


def _is_unknown_character_question(question: str, resolved: str | None) -> bool:
    if resolved:
        return False
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

def _retrieve(
    question: str,
    character_filter: str | None,
    section_filter:   str | None,
) -> tuple[list[str], list[Source]]:
    """
    Search Pinecone. Applies character and optional section filters.

    When a section filter is active we run two queries:
    1. Filtered (character + section) — top-3, most precise
    2. Character-only — top-3, broader context
    The results are merged, deduplicated, and capped at RETRIEVAL_TOP_K.
    This ensures the most relevant section ranks first without completely
    excluding supporting context from other sections.
    """
    def _search(top_k: int, char: str | None, section: str | None) -> list[dict]:
        f: dict = {}
        if char:    f["character"] = {"$eq": char}
        if section: f["section"]   = {"$eq": section}
        query = SearchQuery(
            inputs={"text": question},
            top_k=top_k,
            filter=f if f else None,
        )
        results = pinecone_idx.search(
            namespace=NAMESPACE,
            query=query,
            fields=["text", "character", "section", "last_updated"],
        )
        return results.to_dict().get("result", {}).get("hits", [])

    if section_filter and character_filter:
        # Precise hits first, then broader character context
        hits_focused = _search(2, character_filter, section_filter)
        hits_broad   = _search(3, character_filter, None)
        # Merge, dedup by record id, cap at TOP_K
        seen: set[str] = set()
        hits: list[dict] = []
        for h in hits_focused + hits_broad:
            rid = h.get("_id", "")
            if rid not in seen:
                seen.add(rid)
                hits.append(h)
                if len(hits) >= RETRIEVAL_TOP_K:
                    break
    else:
        hits = _search(RETRIEVAL_TOP_K, character_filter, None)

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

def _build_prompt(question: str, chunks: list[str], sources: list[Source]) -> str:
    """
    Build a labelled context block so Gemini can cite sources accurately
    without having to infer them from content alone.
    """
    context_parts: list[str] = []
    for chunk, source in zip(chunks, sources):
        label = f"[{source.character.title()} — {source.section}]"
        context_parts.append(f"{label}\n{chunk}")

    context = "\n\n---\n\n".join(context_parts)
    return f"Retrieved context:\n\n{context}\n\n---\n\nQuestion: {question}"


def _extract_text(resp) -> str:
    """
    Safely pull text from a Gemini response.

    ``resp.text`` can be None when the model stops on MAX_TOKENS or a safety
    filter, which would otherwise crash Pydantic validation and surface as a
    "Failed to fetch" in the browser. This gathers any partial text and
    falls back to a clear message rather than returning None.
    """
    # Primary path
    text = getattr(resp, "text", None)
    if text:
        return text.strip()

    # Fallback: stitch together any text parts from the first candidate
    try:
        parts = resp.candidates[0].content.parts
        collected = "".join(getattr(p, "text", "") or "" for p in parts)
        if collected.strip():
            return collected.strip()
    except Exception:
        pass

    return (
        "I wasn't able to generate a complete answer for that. "
        "Try rephrasing your question or naming a specific character."
    )


def _call_gemini(contents: str, max_tokens: int) -> str:
    """
    Call Gemini with automatic retry on transient 503 (high demand) errors.

    Google's flash models occasionally return 503 UNAVAILABLE during demand
    spikes. These are temporary and almost always succeed on retry, so we
    back off briefly and try again before giving up.
    """
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = gemini_client.models.generate_content(
                model    = GENERATION_MODEL,
                contents = contents,
                config   = gtypes.GenerateContentConfig(
                    system_instruction = _SYSTEM_INSTRUCTION,
                    temperature        = 0.2,
                    max_output_tokens  = max_tokens,
                ),
            )
            return _extract_text(resp)
        except Exception as exc:
            last_exc = exc
            # Only retry on transient server errors (503 / UNAVAILABLE / overloaded)
            msg = str(exc).lower()
            transient = ("503" in msg or "unavailable" in msg
                         or "overloaded" in msg or "high demand" in msg)
            if transient and attempt < 2:
                time.sleep(1.5 * (attempt + 1))   # 1.5s, then 3s
                continue
            raise
    # Unreachable, but keeps type checkers happy
    raise last_exc  # type: ignore[misc]


def _generate(question: str, chunks: list[str], sources: list[Source]) -> str:
    return _call_gemini(_build_prompt(question, chunks, sources), max_tokens=1200)


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

    # 1. Exact cache check (Redis — fastest path, ~1 ms)
    key        = _cache_key(normalized)
    cached_raw = redis_client.get(key)
    if cached_raw:
        payload = json.loads(cached_raw)
        return QueryResponse(
            answer     = payload["answer"],
            sources    = [Source(**s) for s in payload["sources"]],
            cached     = True,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 2. Semantic cache check (Pinecone — catches rephrased questions, ~50 ms)
    sem_hit = _semantic_cache_lookup(question)
    if sem_hit:
        answer, source_dicts = sem_hit
        sources = [Source(**s) for s in source_dicts]
        # Also write to Redis so the next identical hit is instant
        redis_client.setex(key, REDIS_TTL, json.dumps({
            "answer":  answer,
            "sources": [s.model_dump() for s in sources],
        }))
        return QueryResponse(
            answer     = answer,
            sources    = sources,
            cached     = True,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 3. Character resolution + fallback guard
    char_filter = resolve_character(question)
    if _is_unknown_character_question(question, char_filter):
        return QueryResponse(
            answer     = _fallback_message(question),
            sources    = [],
            cached     = False,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 4. Section intent detection
    section_filter = _detect_section(question)

    # 5. Retrieve
    chunks, sources = _retrieve(question, char_filter, section_filter)

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

    # 6. Generate — never let a Gemini error 500 the endpoint; return a clean
    #    message so the frontend shows it instead of "Failed to fetch".
    try:
        answer = _generate(question, chunks, sources)
    except Exception as exc:
        import traceback
        print("=== GENERATION ERROR ===")
        traceback.print_exc()
        print("========================")
        return QueryResponse(
            answer = (
                "The answer service hit an error while generating a response:\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "Check the server console for the full traceback."
            ),
            sources    = sources,
            cached     = False,
            latency_ms = round((time.monotonic() - t0) * 1000, 1),
        )

    # 7. Cache — write to both Redis (exact match) and Pinecone (semantic)
    redis_client.setex(key, REDIS_TTL, json.dumps({
        "answer":  answer,
        "sources": [s.model_dump() for s in sources],
    }))
    _semantic_cache_store(question, answer, sources)

    return QueryResponse(
        answer     = answer,
        sources    = sources,
        cached     = False,
        latency_ms = round((time.monotonic() - t0) * 1000, 1),
    )


# ---------------------------------------------------------------------------
# Compare endpoint
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    character_a: str
    character_b: str
    aspect:      str = "overall"   # e.g. "artifacts", "teams", "weapons"


class CompareResponse(BaseModel):
    answer:     str
    sources:    list[Source]
    latency_ms: float


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    """
    Compare two characters across a given aspect (default: overall).

    Retrieves context for both characters independently, then asks Gemini
    to synthesize a side-by-side comparison grounded in both chunks.
    """
    t0 = time.monotonic()

    char_a = resolve_character(req.character_a)
    char_b = resolve_character(req.character_b)

    if not char_a:
        raise HTTPException(status_code=404, detail=f"Unknown character: {req.character_a!r}")
    if not char_b:
        raise HTTPException(status_code=404, detail=f"Unknown character: {req.character_b!r}")

    aspect = req.aspect.lower().strip()
    # Map friendly aspect names to section keys
    aspect_map = {
        "artifact": "artifacts", "artifacts": "artifacts",
        "weapon":   "weapons",   "weapons":   "weapons",
        "team":     "teams",     "teams":     "teams",
        "talent":   "talents",   "talents":   "talents",
        "const":    "constellations", "constellation": "constellations",
        "material": "materials", "materials": "materials",
        "overview": "overview",  "overall":   None,
    }
    section = aspect_map.get(aspect)

    question = f"Compare {req.character_a} and {req.character_b}: {aspect}"

    # Fetch chunks for each character separately
    chunks_a, sources_a = _retrieve(question, char_a, section)
    chunks_b, sources_b = _retrieve(question, char_b, section)

    all_chunks  = chunks_a  + chunks_b
    all_sources = sources_a + sources_b

    if not all_chunks:
        raise HTTPException(status_code=404, detail="No data found for one or both characters.")

    # Build a compare-specific prompt
    context_a = "\n\n".join(f"[{char_a.title()} — {s.section}]\n{c}" for c, s in zip(chunks_a, sources_a))
    context_b = "\n\n".join(f"[{char_b.title()} — {s.section}]\n{c}" for c, s in zip(chunks_b, sources_b))
    prompt = (
        f"Character A context:\n\n{context_a}\n\n"
        f"---\n\n"
        f"Character B context:\n\n{context_b}\n\n"
        f"---\n\n"
        f"Question: Compare {req.character_a} and {req.character_b} in terms of {aspect}. "
        f"Use the context above only. Structure your answer with a section for each character, "
        f"then a brief summary of the key differences."
    )

    try:
        answer = _call_gemini(prompt, max_tokens=1500)
    except Exception as exc:
        answer = (
            "The answer service hit an error while generating the comparison:\n\n"
            f"`{type(exc).__name__}: {exc}`\n\nPlease try again in a moment."
        )

    return CompareResponse(
        answer     = answer,
        sources    = all_sources,
        latency_ms = round((time.monotonic() - t0) * 1000, 1),
    )


# ---------------------------------------------------------------------------
# Character list endpoint
# ---------------------------------------------------------------------------

@router.get("/characters")
def list_characters(element: str | None = None, weapon: str | None = None) -> dict:
    """
    List all known characters, optionally filtered by element or weapon type.

    Examples:
        GET /characters
        GET /characters?element=Pyro
        GET /characters?weapon=Claymore
        GET /characters?element=Cryo&weapon=Sword
    """
    from pathlib import Path
    import json as _json

    char_dir = Path("character_data")
    results: list[dict] = []

    # Build from JSON files so we have element/weapon data
    seen: set[str] = set()
    for f in sorted(char_dir.glob("*.json")):
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            name = data.get("character_name")
            if not name or name in seen:
                continue
            seen.add(name)

            ov = data.get("overview", {})
            char_element = ov.get("element", "")
            char_weapon  = ov.get("weapon_type", "")
            char_rarity  = ov.get("rarity")
            char_role    = ov.get("role", "")

            if element and char_element.lower() != element.lower():
                continue
            if weapon and char_weapon.lower() != weapon.lower():
                continue

            results.append({
                "name":        name,
                "element":     char_element,
                "weapon_type": char_weapon,
                "rarity":      char_rarity,
                "role":        char_role,
            })
        except Exception:
            pass

    return {
        "count":      len(results),
        "characters": results,
    }

