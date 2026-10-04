"""
src.api.app
===========
FastAPI application factory.

Importing this module creates the ``app`` instance used by uvicorn::

    uvicorn src.api.app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, query

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Genshin Build Assistant",
    description = "RAG-powered character build and team comp assistant for Genshin Impact.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow all origins — fine for a local personal tool.
# Tighten to specific origins before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(query.router,  tags=["query"])
app.include_router(health.router, tags=["ops"])
