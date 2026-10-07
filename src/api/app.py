"""
src.api.app
===========
FastAPI application factory.

Importing this module creates the ``app`` instance used by uvicorn::

    uvicorn src.api.app:app --reload --port 8000
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .limiter import limiter
from .routers import health, query

# ---------------------------------------------------------------------------
# Logging — structured, single-line records for request observability.
# Configured here (import time) so it applies before any router handles a
# request. Level is controllable via the LOG_LEVEL env var.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
