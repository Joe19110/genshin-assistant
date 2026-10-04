"""
serve.py — root entry point for the API server.

Convenience launcher for development.  For production use uvicorn directly:

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Development usage:

    py serve.py           # auto-reload on code changes
    py serve.py --port 9000
"""

import argparse
import uvicorn

def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Genshin Build Assistant API.")
    parser.add_argument("--host",   default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true",  help="Disable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.app:app",
        host   = args.host,
        port   = args.port,
        reload = not args.no_reload,
    )

if __name__ == "__main__":
    main()
