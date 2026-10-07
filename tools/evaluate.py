"""
tools/evaluate.py — RAGAS-style evaluation of the RAG pipeline.

Computes the two metrics RAGAS is best known for, adapted to this project:

  retrieval_precision  — of the chunks retrieved for a question, what fraction
                         belong to a section the question is actually about?
                         (deterministic: compares retrieved sections vs the
                         eval dataset's expected_sections)

  faithfulness         — is the generated answer supported by the retrieved
                         context, with no invented claims? Judged by Gemini on
                         a 0–1 scale.

  fallback_accuracy    — for 'unknown' characters, did the assistant correctly
                         refuse instead of hallucinating a build?

Run from the project root:
    py tools/evaluate.py                     # default pipeline settings
    py tools/evaluate.py --top-k 3           # tuning iteration (override top-k)
    py tools/evaluate.py --out results_a.json

The script writes a JSON results file and prints a summary table. Run it twice
with different --top-k values to produce the before/after tuning comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types as gtypes

from src.api import dependencies as deps
from src.api.routers import query as q

DATASET_PATH = PROJECT_ROOT / "tools" / "eval_dataset.json"


# ---------------------------------------------------------------------------
# Faithfulness judge
# ---------------------------------------------------------------------------

_JUDGE_INSTRUCTION = """\
You are a strict evaluation judge for a retrieval-augmented QA system.
Given a QUESTION, the retrieved CONTEXT, and the system's ANSWER, decide how
faithful the answer is to the context.

Score from 0.0 to 1.0:
- 1.0  Every claim in the answer is directly supported by the context.
- 0.5  Mostly supported, but contains at least one claim not in the context.
- 0.0  Key claims are unsupported or contradicted by the context.

Respond with ONLY a JSON object: {"score": <float>, "reason": "<one sentence>"}
"""


def _judge_faithfulness(question: str, context: str, answer: str) -> tuple[float, str]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"Return the JSON verdict."
    )
    try:
        resp = deps.gemini_client.models.generate_content(
            model=deps.GENERATION_MODEL,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=_JUDGE_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=200,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(resp.text)
        return float(data.get("score", 0.0)), str(data.get("reason", ""))
    except Exception as exc:
        return 0.0, f"judge error: {exc}"


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def _eval_known(item: dict) -> dict:
    """Evaluate a question about a character we have data for."""
    question = item["question"]
    char = q.resolve_character(question)
    section = q._detect_section(question)

    chunks, sources = q._retrieve(question, char, section)

    # Retrieval precision: fraction of retrieved chunks in an expected section
    expected = set(item["expected_sections"])
    retrieved_sections = [s.section for s in sources]
    if retrieved_sections:
        hits = sum(1 for s in retrieved_sections if s in expected)
        precision = hits / len(retrieved_sections)
    else:
        precision = 0.0

    # Whether the single top-ranked chunk is in an expected section
    top_hit = bool(retrieved_sections and retrieved_sections[0] in expected)

    # Faithfulness: generate an answer and judge it
    if chunks:
        answer = q._generate(question, chunks, sources)
        context = "\n\n".join(chunks)
        faith, reason = _judge_faithfulness(question, context, answer)
    else:
        answer, faith, reason = "", 0.0, "no chunks retrieved"

    return {
        "id": item["id"],
        "question": question,
        "type": "known",
        "resolved_character": char,
        "detected_section": section,
        "retrieved_sections": retrieved_sections,
        "retrieval_precision": round(precision, 3),
        "top1_correct": top_hit,
        "faithfulness": round(faith, 3),
        "faithfulness_reason": reason,
        "answer": answer,
    }


def _eval_unknown(item: dict) -> dict:
    """Evaluate the known-character fallback for an unavailable character."""
    question = item["question"]
    char = q.resolve_character(question)
    is_unknown = q._is_unknown_character_question(question, char)
    # Correct behaviour: resolve to None AND flagged as unknown → fallback fires
    correct = (char is None) and is_unknown

    return {
        "id": item["id"],
        "question": question,
        "type": "unknown",
        "resolved_character": char,
        "fallback_fired": correct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override RETRIEVAL_TOP_K for a tuning run.")
    parser.add_argument("--out", default=None,
                        help="Write results JSON to this path.")
    args = parser.parse_args()

    if args.top_k is not None:
        q.RETRIEVAL_TOP_K = args.top_k
        deps.RETRIEVAL_TOP_K = args.top_k
        print(f"Overriding RETRIEVAL_TOP_K = {args.top_k}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    results: list[dict] = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['id']}: {item['question'][:55]}...")
        if item["character"] == "unknown":
            results.append(_eval_unknown(item))
        else:
            results.append(_eval_known(item))
        time.sleep(0.3)  # gentle pacing for the judge calls

    # ── Aggregate ──
    known   = [r for r in results if r["type"] == "known"]
    unknown = [r for r in results if r["type"] == "unknown"]

    avg_precision = sum(r["retrieval_precision"] for r in known) / len(known) if known else 0.0
    top1_rate     = sum(1 for r in known if r["top1_correct"]) / len(known) if known else 0.0
    avg_faith     = sum(r["faithfulness"] for r in known) / len(known) if known else 0.0
    fallback_rate = sum(1 for r in unknown if r["fallback_fired"]) / len(unknown) if unknown else 0.0

    summary = {
        "top_k": q.RETRIEVAL_TOP_K,
        "n_known": len(known),
        "n_unknown": len(unknown),
        "retrieval_precision": round(avg_precision, 3),
        "top1_accuracy": round(top1_rate, 3),
        "faithfulness": round(avg_faith, 3),
        "fallback_accuracy": round(fallback_rate, 3),
    }

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<22} {v}")

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "tools" / f"eval_results_k{q.RETRIEVAL_TOP_K}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
