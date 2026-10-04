"""
src.pipeline
============
Chunks character JSON data into plain-text passages and upserts them into
the Pinecone vector index with server-side embedding.

Public API
----------
ingest_character(index, path, force)  upsert one character file
get_or_create_index(pc)               return (or create) the Pinecone index
"""

from .ingest import ingest_character, get_or_create_index

__all__ = ["ingest_character", "get_or_create_index"]
