"""
src.api.limiter
===============
Shared rate-limiter instance.

Kept in its own module so routers can apply ``@limiter.limit(...)``
decorators without importing the FastAPI app (which would create a circular
import, since the app imports the routers).

Limits are keyed by client IP and configurable via env vars:

    RATE_LIMIT_QUERY    (default "30/minute")   — /query and /query/stream
    RATE_LIMIT_COMPARE  (default "15/minute")   — /compare
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

RATE_LIMIT_QUERY   = os.environ.get("RATE_LIMIT_QUERY",   "30/minute")
RATE_LIMIT_COMPARE = os.environ.get("RATE_LIMIT_COMPARE", "15/minute")
