"""
Genshin Build Assistant
=======================
RAG-powered character build and team comp assistant for Genshin Impact.

Sub-packages
------------
src.scraper   — scrape game8.co character pages → character_data/*.json
src.pipeline  — chunk JSON → embed → upsert into Pinecone
src.api       — FastAPI application (query, health, cache endpoints)
"""
