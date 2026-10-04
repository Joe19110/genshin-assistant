# Genshin Build & Team Comp Assistant — Technical Spec

Solo, individually-owned RAG project. Solves a real recurring problem (manually searching artifact/team/material guides per character) and closes the portfolio's biggest remaining gap: a fully solo, evaluated AI/data project with zero ownership ambiguity.

**Positioning:** "A build assistant that stays current with patches, is fast on common questions, and refuses to guess on characters it doesn't actually know."

---

## 1. Scope

**In scope:** character build recommendations (artifact sets, main/substat priorities, weapon recommendations), team comp suggestions per character, talent/ascension material requirements, constellation ratings.

**Out of scope (v1):** lore/quest text, farming/domain rotation schedules.

---

## 2. Data source

**Source:** game8.co character build pages. Decision made and confirmed — not revisiting.

**Ethics note (personal reference):** game8.co's robots.txt blocks `GPTBot` and `Google-Extended` specifically but does not blanket-disallow all crawlers. This is a personal, non-commercial, non-published project — scraped data stays local and the app synthesizes answers in its own words rather than republishing verbatim.

**Scraper config:** `requests` + `BeautifulSoup`, 1.5 s delay between requests, custom identifying User-Agent.

**Per-character JSON schema:**
```
character_name, source_url, last_updated
overview       – identity, tier, role, strengths/weaknesses
artifacts      – sets (with piece counts for 2-pc combos), stat priority per slot,
                 goal stat targets
weapons        – ranked list (BiS / Statstick / Best 4-Star / F2P) with explanations
talents        – priority order with rationale, passives labeled combat vs utility
teams          – mono-element comps with per-member roles, flex teammates
constellations – C1–C6 with effects, star ratings, pull recommendation
materials      – ascension + talent items grouped by type
```

**Excluded by design:** voice actor info, release date / how-to-obtain.

---

## 3. Architecture

```
game8.co
  └─ src/scraper/page_parser.py     parse one character page → dict
  └─ src/scraper/runner.py          discover all URLs + bulk scrape loop
       │
       ▼
character_data/*.json               one file per character (gitignored)
       │
       ▼
src/pipeline/flatteners.py          JSON section → plain-text passage (7 flatteners)
src/pipeline/ingest.py              change detection → upsert to Pinecone
       │
       ▼
Pinecone Serverless index           server-side llama-text-embed-v2
       │
       ▼
src/api/routers/query.py            embed question → search → generate
src/api/dependencies.py             Redis cache + character registry
       │
       ▼
frontend/index.html                 vanilla chat UI
```

**Chunking strategy:** one chunk per JSON section per character (7 chunks/character). Queries map naturally to categories (artifacts, weapons, teams, etc.), which keeps retrieval precision easier to reason about than arbitrary token-based splitting.

**Embeddings:** Pinecone integrated server-side `llama-text-embed-v2`. No vectors pass through application code — Pinecone embeds on write (`upsert_records`) and on read (`search(inputs={"text": ...})`).

**Vector store:** Pinecone Serverless, free Starter plan. `genshin-assistant` index, namespace `characters`, `field_map={"text": "text"}`.

**Backend:** FastAPI + Uvicorn. Run locally via `py serve.py` or `uvicorn src.api.app:app --reload`.

**Generation:** Gemini 2.0 Flash (`gemini-2.0-flash`) via `google-genai` SDK. Temperature 0.2, max 1024 output tokens.

**Cache:** Redis 7 (local Docker). Cache key = `"genshin::" + SHA-256(normalized_question)`. TTL 24 h (configurable via `REDIS_TTL` env var).

---

## 4. RAG pipeline (POST /query)

1. Normalize question → check Redis cache → return immediately on hit
2. Resolve character name from question → look up Pinecone metadata key
3. Known-character fallback: if question names a proper noun not in the dataset, return explicit fallback without calling LLM
4. Search Pinecone, top-k=5, optionally filtered by `character` metadata field
5. Guard against empty retrieval (return descriptive message, not an LLM hallucination)
6. Generate with Gemini: grounding system instruction forbids answering outside context, requires citing character/section
7. Write answer + sources to Redis
8. Return `{ answer, sources, cached, latency_ms }`

**System instruction key rules:**
- Answer ONLY from retrieved context — no outside knowledge
- Cite source character and section in every answer
- If context insufficient, say so explicitly — do not guess
- Quote stat values and item names exactly as given

---

## 5. Feature: patch-currency

Genshin patches every ~6 weeks. Re-scrape and selective re-ingest after each patch:

1. `py scrape.py` — scrapes the game8 index page to discover all character URLs, then scrapes each one. Skips characters whose file was already written today (mtime check).
2. `py ingest.py` — before upserting each character, queries Pinecone for the stored `last_updated` value. Skips if unchanged.
3. `DELETE /cache/{character}` — flush Redis after a re-ingest so stale answers don't persist.

This is selective re-processing, not a full re-embed of everything — only changed characters get new vectors.

---

## 6. Feature: answer caching

Redis cache keyed on a SHA-256 hash of the normalized question (lowercased, punctuation stripped, whitespace collapsed). Default TTL 24 h.

Cache invalidation: `DELETE /cache/{character}` flushes the entire Redis DB (acceptable for a single-user tool — surgical per-character deletion would require a secondary question→character index).

For the portfolio write-up: measure actual latency numbers with cache hit vs. miss (target: ~80 ms hit vs. ~1.5 s miss).

---

## 7. Feature: known-character fallback

**Problem:** a question about a character not yet scraped (new release, scrape failure) would cause the LLM to invent a build.

**Implementation:** `src/api/dependencies.py` builds a character registry at startup from `character_data/*.json`. Each character gets up to three lookup keys: full lowercased name, first word (primary), and last word (alias). `resolve_character(question)` scans longest-key-first. If the question names a capitalised proper noun that isn't in the registry, the fallback fires before any Pinecone or LLM call.

**Pytest suite (planned):**
- Known character + valid question → grounded answer with sources
- Unknown/unreleased character → explicit fallback, no LLM call
- Cache hit → cached answer returned, no new LLM call (assert on call count)
- Malformed/empty question → 400 with clear message

---

## 8. File map

```
genshin-assistant/
│
├── scrape.py                       entry point → src.scraper.runner
├── ingest.py                       entry point → src.pipeline.ingest
├── serve.py                        entry point → src.api.app
│
├── src/
│   ├── __init__.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── page_parser.py          parse one game8 page → structured dict
│   │   └── runner.py               URL discovery + bulk scrape loop + CLI
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── flatteners.py           7 section flatteners + FLATTENERS registry
│   │   └── ingest.py               Pinecone index mgmt + change detection + CLI
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py                  FastAPI factory (includes routers)
│       ├── dependencies.py         client singletons + character registry
│       └── routers/
│           ├── __init__.py
│           ├── query.py            POST /query
│           └── health.py           GET /health, DELETE /cache/{name}
│
├── frontend/
│   └── index.html                  vanilla chat UI (no build step)
│
├── character_data/                 scraped JSONs (gitignored)
│
├── docker-compose.yml              Redis 7
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md                       this file
```




## 9. Running the project

```bash
# Install deps
py -m pip install -r requirements.txt

# Configure env
copy .env.example .env   # then fill GEMINI_API_KEY and PINECONE_API_KEY

# Start Redis
docker compose up -d

# Scrape characters (first time or after a patch)
py scrape.py              # all ~120 characters, ~3 min
py scrape.py --limit 5   # test with 5 first
py scrape.py --dry-run   # see URLs without scraping

# Ingest into Pinecone
py ingest.py              # all characters (skips unchanged)
py ingest.py -c varka     # single character
py ingest.py --force      # bypass change detection

# Start the API
py serve.py               # auto-reload dev server on :8000
# or: uvicorn src.api.app:app --reload --port 8000

# Open the frontend
# Open frontend/index.html in a browser directly — no server needed.

# Post-patch cache flush
curl -X DELETE http://localhost:8000/cache/varka
```

---

## 10. API reference

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | RAG pipeline. Body: `{"question": "..."}` |
| `/health` | GET | Liveness check + known characters + Redis status |
| `/cache/{name}` | DELETE | Flush Redis cache (post-patch workflow) |
| `/docs` | GET | FastAPI auto-generated Swagger UI |

`/query` response shape:
```json
{
  "answer":     "...",
  "sources":    [{ "character": "varka", "section": "artifacts", "score": 0.91, "last_updated": "2026-08-26" }],
  "cached":     false,
  "latency_ms": 1240.3
}
```

---

## 11. Evaluation (planned)

- ~15–20 realistic test questions with known-correct answers
- RAGAS for faithfulness and retrieval-precision scores
- One documented tuning iteration (chunk size, k, or prompt wording)
- Cache hit/miss latency numbers measured and recorded

---

## 12. Stack summary

| Layer | Choice | Rationale |
|---|---|---|
| Embeddings | Pinecone integrated (llama-text-embed-v2) | No vectors in code; free tier; simplest architecture |
| Generation | Gemini 2.0 Flash | Free tier, fast, consistent with Gemini ecosystem |
| Vector DB | Pinecone Serverless | Already familiar from BINUS EDM DESA; free tier sufficient |
| Cache | Redis 7 (Docker) | Local, zero cost, instant setup |
| API | FastAPI | Async, typed, auto-docs |
| Frontend | Vanilla HTML/JS | No build step; easy to extend |

---

## 13. Known issues / notes

- **`last_updated`** in the JSON reflects game8's content revision date (scraped from `archive_revision=YYYYMMDDHHMMSS` in a tracking pixel URL), not the date we scraped it. The ingest change detection correctly compares this value against Pinecone metadata — don't confuse it with file mtime, which is used by the scraper's skip logic.

- **Character name variants** in the character registry (e.g. `"tartaglia"` → `"tartaglia (childe)"`, `"heizou"` → `"shikanoin heizou"`) are generated from first-word and last-word extraction at startup. The `SKIP_PRIMARIES` set in `dependencies.py` excludes single-word primaries that would cause false positives on element names (`cryo`, `pyro`) or incomplete two-word names (`hu`, `lan`, `yun`).

- **Pinecone SDK v7.0.2 API differences** from the docs (written for a newer version):
  - `pc.list_indexes().names()` not `pc.indexes.list()`
  - `pc.create_index_for_model()` not `pc.indexes.create_for_model()`
  - `pc.Index(name)` not `pc.index(name)`
  - `index.search(namespace, query=SearchQuery(inputs, top_k, filter), fields)` — `top_k` etc. are on the `SearchQuery` object, not top-level kwargs
  - Response: `result.to_dict()["result"]["hits"]`, each hit has `_id`, `_score`, `fields`
