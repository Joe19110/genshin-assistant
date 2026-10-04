"""
src.scraper.runner
==================
Discovers all playable character build URLs from game8's "All Best Builds"
index page and bulk-scrapes them into character_data/.

CLI usage
---------
    py -m src.scraper.runner                 # scrape all (skips today's files)
    py -m src.scraper.runner --force         # re-scrape everything
    py -m src.scraper.runner --dry-run       # print discovered URLs, no scraping
    py -m src.scraper.runner --limit 5       # first N characters only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .page_parser import scrape_character_page, HEADERS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_URL = "https://game8.co/games/Genshin-Impact/archives/530535"
OUTPUT_DIR = Path("character_data")
REQUEST_DELAY_SECONDS = 1.5

# Characters confirmed on the index page as playable
KNOWN_CHAR_NAMES: frozenset[str] = frozenset({
    "aino", "albedo", "alhaitham", "aloy", "amber", "arlecchino", "ayaka",
    "ayato", "baizhu", "barbara", "beidou", "bennett", "candace", "charlotte",
    "chasca", "chevreuse", "chiori", "chongyun", "citlali", "clorinde",
    "collei", "columbina", "cyno", "dahlia", "dehya", "diluc", "diona",
    "dori", "durin", "emilie", "escoffier", "eula", "faruzan", "fischl",
    "flins", "freminet", "furina", "gaming", "ganyu", "gorou", "heizou",
    "hu tao", "iansan", "ifa", "illuga", "ineffa", "itto", "jahoda", "jean",
    "kachina", "kaeya", "kaveh", "kazuha", "keqing", "kinich", "kirara",
    "klee", "kokomi", "lan yan", "lauma", "layla", "linnea", "lisa", "lohen",
    "lynette", "lyney", "mavuika", "mika", "mizuki", "mona", "mualani",
    "nahida", "navia", "nefer", "neuvillette", "nicole", "nilou", "ningguang",
    "noelle", "odette", "ororon", "prune", "qiqi", "raiden", "razor",
    "rosaria", "sandrone", "sara", "sayu", "sethos", "shenhe", "shinobu",
    "sigewinne", "skirk", "sucrose", "tartaglia", "thoma", "tighnari",
    "varesa", "varka", "venti", "wanderer", "wriothesley", "xiangling",
    "xianyun", "xiao", "xilonen", "xingqiu", "xinyan", "yae miko", "yanfei",
    "yaoyao", "yelan", "yoimiya", "yun jin", "zhongli", "zibai",
    "traveler (anemo)", "traveler (cryo)", "traveler (dendro)",
    "traveler (electro)", "traveler (geo)", "traveler (hydro)", "traveler (pyro)",
})

# Upcoming characters — no build page exists yet
UPCOMING_SKIP: frozenset[str] = frozenset({
    "alice", "capitano", "dainsleif", "danica", "dottore", "mitya",
    "noy", "pantalone", "pierro", "pulcinella", "rerir", "tsaritsa",
    "valeriy", "vesna", "vodyanitsa",
})

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_character_urls() -> list[tuple[str, str]]:
    """
    Return ``[(character_name, url), ...]`` for all playable characters
    listed on the game8 "All Best Builds" index page.
    """
    print(f"Fetching character index: {INDEX_URL}")
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    results: list[tuple[str, str]] = []

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "/games/Genshin-Impact/archives/" not in href:
            continue

        text = a.get_text(strip=True)
        if not text or text.lower() not in KNOWN_CHAR_NAMES:
            continue

        full_url = "https://game8.co" + href if href.startswith("/") else href
        char_key = text.lower()

        if full_url in seen_urls or char_key in seen_names:
            continue
        if char_key.split()[0] in UPCOMING_SKIP:
            continue

        seen_urls.add(full_url)
        seen_names.add(char_key)
        results.append((text, full_url))

    return results


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def _already_scraped_today(char_name: str) -> bool:
    """
    True if a JSON for *char_name* was written today.

    Uses file mtime rather than ``last_updated`` (which reflects game8's
    content revision date, not our scrape date).

    Matching is first-word fuzzy: ``"Tartaglia"`` matches
    ``tartaglia_(childe).json`` because both start with ``tartaglia``.
    """
    first_word = char_name.lower().split()[0]
    today = date.today()
    for path in OUTPUT_DIR.glob("*.json"):
        if path.stem.lower().startswith(first_word):
            if date.fromtimestamp(path.stat().st_mtime) == today:
                return True
    return False


# ---------------------------------------------------------------------------
# Scrape loop
# ---------------------------------------------------------------------------

def scrape_all(
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    characters = discover_character_urls()
    print(f"Discovered {len(characters)} character build pages.\n")

    if limit:
        characters = characters[:limit]
        print(f"Limited to first {limit} characters.\n")

    if dry_run:
        for name, url in characters:
            print(f"  {name:<30} {url}")
        return

    success = skipped = 0
    failed: list[tuple[str, str, str]] = []

    for i, (char_name, url) in enumerate(characters, 1):
        prefix = f"[{i}/{len(characters)}]"

        if not force and _already_scraped_today(char_name):
            print(f"{prefix} SKIP  {char_name}")
            skipped += 1
            continue

        print(f"{prefix} Scraping {char_name} ...")
        try:
            data = scrape_character_page(url)
            fname = (data.get("character_name") or char_name).lower().replace(" ", "_")
            out_path = OUTPUT_DIR / f"{fname}.json"
            out_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"         → {out_path}")
            success += 1
        except Exception as exc:
            print(f"         ERROR: {exc}")
            failed.append((char_name, url, str(exc)))

        if i < len(characters):
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{'='*50}")
    print(f"Done. {success} scraped, {skipped} skipped, {len(failed)} failed.")
    if failed:
        print("\nFailed:")
        for name, url, err in failed:
            print(f"  {name}: {err}")
            print(f"    {url}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-scrape all Genshin character build pages from game8."
    )
    parser.add_argument("--force",   action="store_true", help="Re-scrape even if JSON exists today.")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without scraping.")
    parser.add_argument("--limit",   type=int,            help="Only scrape first N characters.")
    args = parser.parse_args()
    scrape_all(force=args.force, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
