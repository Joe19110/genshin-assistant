"""
src.pipeline.ingest
===================
Chunks character JSON files into plain-text passages and upserts them into
the Pinecone index with server-side embedding (llama-text-embed-v2).

Change detection
----------------
Before upserting, we compare the character's ``last_updated`` field (game8
content revision date) against what is stored in Pinecone metadata.
Characters whose data hasn't changed since the last ingest are skipped.

Pinecone SDK notes (v7.0.2)
----------------------------
- ``pc.list_indexes().names()``          → list[str]
- ``pc.create_index_for_model(...)``     → creates integrated-embedding index
- ``pc.Index(name)``                     → Index handle
- ``index.upsert_records(namespace, records)``
- ``index.search(namespace, query=SearchQuery(...), fields=[...])``
  Response: ``result.to_dict()["result"]["hits"]``

CLI usage
---------
    py -m src.pipeline.ingest              # ingest all character_data/*.json
    py -m src.pipeline.ingest -c varka     # single character
    py -m src.pipeline.ingest --force      # ignore change detection
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, IndexEmbed
from pinecone.db_data.dataclasses.search_query import SearchQuery

from .flatteners import FLATTENERS

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.environ.get("PINECONE_INDEX_NAME", "genshin-assistant")
PINECONE_CLOUD   = os.environ.get("PINECONE_CLOUD",      "aws")
PINECONE_REGION  = os.environ.get("PINECONE_REGION",     "us-east-1")

CHARACTER_DATA_DIR = Path("character_data")

#: Field name used in both the Pinecone field_map and each upserted record.
#: Must match what the index was created with.
TEXT_FIELD = "text"

#: Namespace that scopes all character chunks in the index.
NAMESPACE = "characters"

#: Polite delay between upsert calls to respect free-tier rate limits.
UPSERT_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone):
    """
    Return a handle to the Pinecone index, creating it if it doesn't exist.

    The index uses Pinecone's integrated server-side embedding so raw vectors
    never pass through application code.
    """
    existing = pc.list_indexes().names()
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed=IndexEmbed(
                model="llama-text-embed-v2",
                field_map={"text": TEXT_FIELD},
                write_parameters={"input_type": "passage"},
                read_parameters={"input_type": "query"},
            ),
        )
        print("Index ready.")
    else:
        print(f"Using existing index '{INDEX_NAME}'.")

    return pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _get_stored_last_updated(index, character_name: str) -> str | None:
    """
    Query Pinecone for the stored ``last_updated`` value of a character.
    Returns ``None`` if the character has never been ingested.
    """
    try:
        results = index.search(
            namespace=NAMESPACE,
            query=SearchQuery(
                inputs={"text": f"{character_name} overview"},
                top_k=1,
                filter={"character": {"$eq": character_name.lower()}},
            ),
            fields=["last_updated", "section"],
        )
        hits = results.to_dict().get("result", {}).get("hits", [])
        # Prefer the overview chunk; fall back to any hit
        for hit in hits:
            if hit.get("fields", {}).get("section") == "overview":
                return hit["fields"].get("last_updated")
        if hits:
            return hits[0].get("fields", {}).get("last_updated")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def ingest_character(index, char_file: Path, force: bool = False) -> int:
    """
    Chunk, embed, and upsert one character JSON file.

    Returns the number of chunks upserted (0 if skipped).
    """
    with open(char_file, encoding="utf-8") as f:
        data = json.load(f)

    char_name    = data.get("character_name") or char_file.stem
    last_updated = data.get("last_updated", "unknown")

    # --- Change detection ---
    if not force:
        stored = _get_stored_last_updated(index, char_name)
        if stored == last_updated:
            print(f"  [{char_name}] unchanged ({last_updated}) — skipping.")
            return 0
        action = f"updated {stored} → {last_updated}" if stored else "new character"
        print(f"  [{char_name}] {action}, ingesting.")

    # --- Build records ---
    records: list[dict] = []
    for section, flattener in FLATTENERS.items():
        if section not in data:
            continue
        try:
            chunk_text = flattener(data)
        except Exception as exc:
            print(f"  [{char_name}] WARNING: could not flatten '{section}': {exc}")
            continue

        record_id = f"{char_name.lower()}::{section}"
        records.append({
            "_id":        record_id,
            TEXT_FIELD:   chunk_text,
            # Metadata stored alongside the vector for filtering and source display
            "character":    char_name.lower(),
            "section":      section,
            "last_updated": last_updated,
            "record_id":    record_id,
        })

    # --- Upsert ---
    if records:
        index.upsert_records(namespace=NAMESPACE, records=records)
        time.sleep(UPSERT_DELAY_S)
        print(f"  [{char_name}] upserted {len(records)} chunks.")

    return len(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Genshin character data into Pinecone."
    )
    parser.add_argument(
        "-c", "--character",
        metavar="NAME",
        help="Ingest only this character (file stem, e.g. 'varka'). "
             "Omit to process all files in character_data/.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Re-embed and upsert even if last_updated hasn't changed.",
    )
    args = parser.parse_args()

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    if args.character:
        target = CHARACTER_DATA_DIR / f"{args.character.lower()}.json"
        if not target.exists():
            print(f"ERROR: {target} not found.")
            return
        files = [target]
    else:
        files = sorted(CHARACTER_DATA_DIR.glob("*.json"))
        if not files:
            print(f"No JSON files in {CHARACTER_DATA_DIR}/. Run the scraper first.")
            return

    total = sum(ingest_character(index, f, force=args.force) for f in files)
    print(f"\nDone. {total} chunk(s) upserted across {len(files)} file(s).")


if __name__ == "__main__":
    main()
