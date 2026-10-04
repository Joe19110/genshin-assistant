"""
ingest.py — root entry point for the ingest pipeline.

Thin wrapper around src.pipeline.ingest so the workflow command stays simple:

    py ingest.py              # ingest all character_data/*.json
    py ingest.py -c varka     # single character
    py ingest.py --force      # bypass change detection
"""

from src.pipeline.ingest import main

if __name__ == "__main__":
    main()
