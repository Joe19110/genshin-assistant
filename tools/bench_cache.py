"""
tools/bench_cache.py — cache latency benchmark.

Measures the payoff of the three-layer cache (Redis exact → Pinecone semantic
→ Gemini generate) by firing each question twice against a running API:

  1st call → cache MISS  (full retrieval + Gemini generation)
  2nd call → cache HIT   (served from Redis, no Gemini call)

It records the server-reported ``latency_ms`` for each, computes the speed-up,
and writes a JSON results file plus a printed summary table.

Prerequisites
-------------
The API must be running and reachable (default http://localhost:8000), with
Pinecone populated and Redis up. Start it with:

    py -m uvicorn src.api.app:app --reload

Then run:

    py tools/bench_cache.py
    py tools/bench_cache.py --url http://localhost:8000 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A spread of questions across characters and sections so the benchmark
# exercises different retrieval paths, not just one cached entry.
BENCH_QUESTIONS: list[str] = [
    "What are the best artifacts for Varka?",
    "What weapon should Neuvillette use?",
    "Who are good teammates for Furina?",
    "What's the talent priority for Raiden Shogun?",
    "Is Hu Tao worth pulling constellations for?",
    "What ascension materials does Nahida need?",
    "How good is Arlecchino overall?",
    "Best artifacts for Yelan?",
]


def _fire(client: httpx.Client, url: str, question: str) -> dict:
    t0 = time.perf_counter()
    resp = client.post(f"{url}/query", json={"question": question}, timeout=60.0)
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    resp.raise_for_status()
    body = resp.json()
    return {
        "server_latency_ms": body.get("latency_ms"),
        "wall_latency_ms":   wall_ms,
        "cached":            body.get("cached"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cache latency.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running API.")
    parser.add_argument("--out", default=None,
                        help="Write results JSON to this path.")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    rows: list[dict] = []

    with httpx.Client() as client:
        # Health check first so a down server fails fast with a clear message.
        try:
            client.get(f"{base}/health", timeout=10.0).raise_for_status()
        except Exception as exc:
            print(f"ERROR: API not reachable at {base} ({exc}).")
            print("Start it with:  py -m uvicorn src.api.app:app --reload")
            raise SystemExit(1)

        for i, question in enumerate(BENCH_QUESTIONS, 1):
            print(f"[{i}/{len(BENCH_QUESTIONS)}] {question[:50]}...")

            miss = _fire(client, base, question)   # 1st call → should miss
            time.sleep(0.2)
            hit  = _fire(client, base, question)   # 2nd call → should hit

            miss_ms = miss["server_latency_ms"] or miss["wall_latency_ms"]
            hit_ms  = hit["server_latency_ms"]  or hit["wall_latency_ms"]
            speedup = round(miss_ms / hit_ms, 1) if hit_ms else None

            rows.append({
                "question":    question,
                "miss_ms":     miss_ms,
                "hit_ms":      hit_ms,
                "miss_cached": miss["cached"],
                "hit_cached":  hit["cached"],
                "speedup_x":   speedup,
            })
            time.sleep(0.2)

    # ── Aggregate ──
    miss_times = [r["miss_ms"] for r in rows if r["miss_ms"]]
    hit_times  = [r["hit_ms"]  for r in rows if r["hit_ms"]]

    summary = {
        "n":                len(rows),
        "miss_ms_mean":     round(statistics.mean(miss_times), 1) if miss_times else None,
        "miss_ms_median":   round(statistics.median(miss_times), 1) if miss_times else None,
        "hit_ms_mean":      round(statistics.mean(hit_times), 1) if hit_times else None,
        "hit_ms_median":    round(statistics.median(hit_times), 1) if hit_times else None,
        "mean_speedup_x":   round(
            statistics.mean([r["speedup_x"] for r in rows if r["speedup_x"]]), 1
        ) if any(r["speedup_x"] for r in rows) else None,
        "hits_served":      sum(1 for r in rows if r["hit_cached"]),
    }

    print("\n" + "=" * 60)
    print("CACHE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"  {'question':<40} {'miss':>7} {'hit':>7} {'x':>5}")
    print("  " + "-" * 60)
    for r in rows:
        q = (r["question"][:37] + "...") if len(r["question"]) > 40 else r["question"]
        print(f"  {q:<40} {r['miss_ms']:>7} {r['hit_ms']:>7} {str(r['speedup_x']):>5}")
    print("  " + "-" * 60)
    for k, v in summary.items():
        print(f"  {k:<18} {v}")

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "tools" / "bench_results.json"
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
