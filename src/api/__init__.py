"""
src.api
=======
FastAPI application for the Genshin build assistant.

Endpoints
---------
POST /query          RAG query pipeline
GET  /health         Liveness check
DELETE /cache/{name} Flush Redis cache for a character
"""
