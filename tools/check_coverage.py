"""
tools/check_coverage.py — scrape every character and report per-field coverage.

A QA / maintenance tool. Scrapes all characters discovered by the runner,
then prints which fields are empty/null for each. Fields are grouped as
CRITICAL (should always be present — empty means a scraper bug) vs
OPTIONAL (genuinely absent on some game8 pages — empty is acceptable).

Run from the project root:
    py tools/check_coverage.py            # all characters (slow, ~3 min)
    py tools/check_coverage.py 20         # first 20 only (quick sample)
"""

import sys
import time
from pathlib import Path

# Resolve the project root (parent of tools/) so imports work from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper.runner import discover_character_urls
from src.scraper.page_parser import scrape_character_page

# Fields that should ALWAYS be present — empty = scraper bug
CRITICAL = {
    "element":        lambda d: d["overview"].get("element"),
    "rarity":         lambda d: d["overview"].get("rarity"),
    "weapon_type":    lambda d: d["overview"].get("weapon_type"),
    "tier_ratings":   lambda d: any(v for v in d["overview"].get("tier_ratings", {}).values()),
    "strengths":      lambda d: d["overview"].get("strengths"),
    "artifacts.sets": lambda d: d["artifacts"].get("sets"),
    "stat.sands":     lambda d: d["artifacts"]["stat_priority"].get("sands"),
    "stat.substats":  lambda d: d["artifacts"]["stat_priority"].get("substats"),
    "weapons":        lambda d: d["weapons"].get("ranked"),
    "teams":          lambda d: d["teams"].get("comps") or d["teams"].get("flex_teammates"),
    "talent.order":   lambda d: d["talents"].get("priority_order"),
    "passives":       lambda d: d["talents"].get("passives"),
    "constellations": lambda d: d["constellations"].get("levels"),
    "mat.gems":       lambda d: d["materials"]["ascension"].get("ascension_gems"),
    "mat.books":      lambda d: d["materials"]["talents"].get("talent_books"),
}

# Fields that are genuinely missing on some pages — empty is acceptable
OPTIONAL = {
    "goal_stats": lambda d: d["artifacts"].get("goal_stats"),
    "four_star":  lambda d: d["artifacts"].get("four_star_option"),
    "weaknesses": lambda d: d["overview"].get("weaknesses"),
    "tier_expl":  lambda d: d["overview"].get("tier_explanation"),
}


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    chars = discover_character_urls()
    if limit:
        chars = chars[:limit]

    print(f"Checking {len(chars)} characters...\n")

    critical_failures: dict[str, list[str]] = {k: [] for k in CRITICAL}
    optional_missing:  dict[str, list[str]] = {k: [] for k in OPTIONAL}
    scrape_errors: list[tuple[str, str]] = []
    perfect = 0

    for i, (name, url) in enumerate(chars, 1):
        try:
            data = scrape_character_page(url)
        except Exception as e:
            scrape_errors.append((name, str(e)))
            print(f"[{i}/{len(chars)}] {name:<22} SCRAPE ERROR: {e}")
            time.sleep(1.5)
            continue

        char_crit = [k for k, check in CRITICAL.items() if not check(data)]
        char_opt  = [k for k, check in OPTIONAL.items() if not check(data)]

        for k in char_crit:
            critical_failures[k].append(name)
        for k in char_opt:
            optional_missing[k].append(name)

        if not char_crit:
            perfect += 1
            print(f"[{i}/{len(chars)}] {name:<22} OK")
        else:
            print(f"[{i}/{len(chars)}] {name:<22} MISSING: {', '.join(char_crit)}")

        time.sleep(1.5)

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"SUMMARY: {perfect}/{len(chars)} characters have all critical fields")
    if scrape_errors:
        print(f"\n{len(scrape_errors)} scrape errors:")
        for name, err in scrape_errors:
            print(f"  {name}: {err}")

    print("\n--- Critical field failures (scraper bugs to fix) ---")
    any_fail = False
    for field, names in critical_failures.items():
        if names:
            any_fail = True
            preview = ", ".join(names[:8]) + (f" +{len(names)-8} more" if len(names) > 8 else "")
            print(f"  {field:<18} {len(names)} chars: {preview}")
    if not any_fail:
        print("  None — all critical fields covered!")

    print("\n--- Optional field gaps (likely genuinely absent on game8) ---")
    for field, names in optional_missing.items():
        if names:
            print(f"  {field:<18} {len(names)} chars missing")


if __name__ == "__main__":
    main()
